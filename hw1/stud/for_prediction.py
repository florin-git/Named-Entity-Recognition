import numpy as np
from typing import List, Any

import torch
from torch import nn
from torch.utils.data import Dataset
from TorchCRF import CRF

from stud.preprocessing import *

class NER_Dataset(Dataset):
    """
    Parameters
    ----------
    sentences: List[List[str]]
        A list of lists of strings where each nested list represents a sentence.
        
    sentences_labels: List[List[str]]
        A list of lists of strings where each nested list represents a sentence,
        containing the labels of the tokens.
        If they are None, then this is a Test dataset.

    vocab: dict
        The map that associates to each token an unique number.
        
    lab2idx: dict
        The map that associates to each label an unique number.

    unk_token: str
        The OOV token.
    """
    def __init__(self, sentences: List[List[str]], vocab: dict, lab2idx: dict, 
                pad_token: str, unk_token: str, sentences_labels: List[List[str]]=None):
        
        # True if the Dataset is for the Test set, where
        # you do not have the labels
        self.test = sentences_labels is None
        
        if not self.test:
            self.labels = sentences_labels
            
            assert len(sentences) == len(sentences_labels), \
                    "Inputs must be of the same length"
            
            self.Y = self._from_sequence_to_idx(sentences_labels, lab2idx)

        # Clean the sentences
        self.sentences = list(map(clean_text, sentences))
        
        self.sentences_lengths = [len(s) for s in sentences]
        self.X = self._from_sequence_to_idx(self.sentences, vocab, unk_token)
            
            
    def _from_sequence_to_idx(self, sequences_list: List[List[str]],
                              vocab: dict, unk_token: str = None) -> List[List[int]]:
        """
        Returns
        -------
            A list of lists of int built by replacing 
            each token with its corresponding id in the vocabulary.
            This is a general function so it works also for labels.
            
        Parameters
        ----------
        sequences_list: List[List[str]]
            A list of lists of strings where each nested list represents a sentence.
            
        vocab: dict
            The map that associates to each token an unique number.

        unk_token: str
            The OOV token.
        """
        
        sequences_idx = []
        
        if unk_token is not None: # For words
            for sentence in sequences_list:
                sequences_idx.append([vocab.get(token, vocab[unk_token]) for token in sentence])
        else: # For labels
            for sentence in sequences_list:
                sequences_idx.append([vocab.get(token) for token in sentence])
        
        return sequences_idx
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        if self.test:
            return self.X[idx], self.sentences_lengths[idx]
        else:
            return self.Y[idx], self.X[idx], self.sentences_lengths[idx]


def pad_sequence(sequence: List[Any], max_length: int, pad_token: str) -> List[Any]:
    """
    Returns
    -------
        A list padded with the 'pad_token' value until 
        it is 'max_length' long. 

    Parameters
    ----------
    sequence: List[Any]
        The list to be padded. 

    max_length: int
        The length of the output list.

    pad_token: str
        The Padding value.
    """
    padded_sequence = [pad_token] * max_length

    for i, token in enumerate(sequence):
        padded_sequence[i] = token

    return padded_sequence

class CollatorPredict(object):
    def __init__(self, vocab, pad_token, device):
        self.vocab = vocab
        self.pad_token = pad_token
        self.device = device
        
    def __call__(self, batch):
        """
        Returns
        -------
            Tensors of the zipped lists in input, sorted in
            descending order by sentences length.
            (The ordering can be useful if 'pack_padded_sequence'
            will be used)

        Parameters
        ----------
        batch
            A zipped python object containing features and labels.

        vocab: dict
            The map that associates to each token an unique number.
        
        pad_token: str
            The Padding token.

        device: str
            Where (CPU/GPU) to load your model.
        """
        features_list = []

        features, sentences_lengths = zip(*batch)

        sorted_batch = sorted(zip(features, sentences_lengths), 
                              key=lambda p: len(p[0]), reverse=True)
        features, sentences_lengths = zip(*sorted_batch)


        max_length_in_batch = np.max(sentences_lengths)

        # Pad sentences and labels to the length of the longest sequence in the batch
        for feature in features:
            features_list.append(pad_sequence(feature, max_length_in_batch, self.vocab[self.pad_token]))

        features_tensor = torch.LongTensor(features_list).to(self.device)

        return features_tensor, sentences_lengths


class NER_Classifier(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        
        # Fasttext
        self.fast_embeddings = self._from_pretrained_embeddings(h_params['fast_embeddings'],
                                                               h_params['vocab_size'],
                                                               h_params['fast_embed_dim'],
                                                               freeze=h_params['freeze_fast'])
            

        # Glove
        self.glove_embeddings = self._from_pretrained_embeddings(h_params['glove_embeddings'],
                                                                 h_params['vocab_size'],
                                                                 h_params['glove_embed_dim'],
                                                                 freeze=h_params['freeze_glove'])
        
        lstm_input_dim = h_params['fast_embed_dim'] + h_params['glove_embed_dim']
        
        # LSTM
        self.lstm = nn.LSTM(lstm_input_dim, 
                            h_params['lstm_hidden_dim'], 
                            bidirectional=h_params['bidirectional'],
                            num_layers=h_params['num_layers'],
                            dropout=h_params['dropout'] if h_params['num_layers'] > 1 else 0,
                            batch_first=True)
        
        
        lstm_output_dim = h_params['lstm_hidden_dim'] if h_params['bidirectional'] is False \
                            else h_params['lstm_hidden_dim'] * 2
        

        self.dropout = nn.Dropout(h_params['dropout'])  

        self.concat = nn.Linear(lstm_output_dim, lstm_output_dim)

        self.classifier = nn.Linear(lstm_output_dim, h_params['num_classes'])
        
        self.relu = nn.LeakyReLU()
        
        if h_params['use_crf']:
            self.crf = CRF(h_params['num_classes'])

        self._init_linear_weights()
        
        
    def forward(self, x, x_lengths):
        x_fast = self.fast_embeddings(x)
        x_glove = self.glove_embeddings(x)
    
        # Concatenate the word embeddings
        x = torch.cat((x_fast, x_glove), dim=2)  
        x = self.dropout(x)
        
        x, _ = self.lstm(x)
        x = self.relu(x)
        
        x = self.concat(x)
        x = self.relu(x)

        output = self.classifier(x)

        return output


    def _from_pretrained_embeddings(self, pretrained_embeddings: torch.Tensor, 
                                    vocab_size: int, embed_dim: int, 
                                    freeze: bool) -> torch.Tensor:
        """
        Returns
        -------
            Pretrained embeddings from input.
            
        Parameters
        ----------
        pretrained_embeddings: torch.Tensor
            Embeddings to be loaded.
            
        vocab_size: int
            Size of the dictionary of embeddings.
            
        embed_dim: int
            Size of each embedding vector.
            
        freeze: bool
            If True the embeddings weights will be not updated
            during training.
        
        """
        
        embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Get emebeddings from pretrained ones
        embeddings.weight.data.copy_(pretrained_embeddings)
        
        # Freeze embeddings
        embeddings.weight.requires_grad = not freeze 
        
        return embeddings
    
    def _init_linear_weights(self):
        initrange = 0.5

        self.concat.weight.data.uniform_(-initrange, initrange)
        self.concat.bias.data.zero_()

        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
