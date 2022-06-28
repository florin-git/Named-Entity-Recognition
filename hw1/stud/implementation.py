import logging
from random import shuffle
import numpy as np
from typing import List, Any, Tuple

from model import Model

# What I add
import torch
from torch.utils.data import DataLoader

from stud.for_prediction import *


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()
    return StudentModel(device)


def flat_list(l: List[List[Any]]) -> List[Any]:
        return [_e for e in l for _e in e]

class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device):

        self.device = device
        self.paths = self._init_paths()
        
        self.vocab = self._read_vocab(self.paths['vocab'])
        self.lab2idx = self._read_vocab(self.paths['lab2idx'])
        self.idx2lab = {idx: label for label, idx in self.lab2idx.items()}

        self.fast_pretrained_embeddings = torch.load(self.paths['fast_pretrained_embeddings'])
        self.glove_pretrained_embeddings = torch.load(self.paths['glove_pretrained_embeddings'])

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"


        self.h_params = {
            'vocab_size': len(self.vocab),
            'fast_embed_dim': 300,
            'freeze_fast': True,
            'glove_embed_dim': 100,
            'freeze_glove': False,
            'lstm_hidden_dim': 512, 
            'num_classes': len(self.lab2idx),
            'fast_embeddings': self.fast_pretrained_embeddings,
            'glove_embeddings': self.glove_pretrained_embeddings,
            'bidirectional': True,
            'num_layers': 3,
            'dropout': 0.4,
            'use_crf': True,  
        }

        self.model = NER_Classifier(self.h_params).to(device)
        self.model.load_state_dict(torch.load(self.paths['model_path'], map_location=device))


    def _read_vocab(self, path: str) -> dict:
        """
        Returns
        -------
            A dictionary that maps tokens to integers.
            
        Parameters
        ----------
        path: str
            Data path of the dictionary.
        """
        vocab = {}
        with open(path, 'r', newline="", encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                vocab[line[0]] = int(line[1])
        return vocab


    def _init_paths(self):
        paths = {
            "vocab": "model/data/vocab.tsv",
            "lab2idx": "model/data/lab2idx.tsv",
            "fast_pretrained_embeddings": "model/pretrained/load_embeddings/pre_fast.pth",
            "glove_pretrained_embeddings": "model/pretrained/load_embeddings/pre_glove.pth",
            "model_path": "model/my_models/my_model_727.pth"
        }
        return paths    


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        
        y_pred_list = []

        dataset = NER_Dataset(tokens, self.vocab, self.lab2idx, self.pad_token, self.unk_token)
        collator = CollatorPredict(self.vocab, self.pad_token, self.device)

        self.model.eval()
        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=1, 
                                collate_fn=collator, shuffle=False)

            for (features, sentences_lengths) in dataloader:
                predictions = self.model(features, sentences_lengths)
                

                # This happens in training
                viterbi_mask = features != self.lab2idx[self.pad_token]
                viterbi_pred = flat_list(self.model.crf.viterbi_decode(predictions, mask=viterbi_mask))


                predictions = predictions.view(-1, predictions.shape[-1])
                viterbi_pred_labels = [self.idx2lab[l] for l in viterbi_pred]
                y_pred_list.append(viterbi_pred_labels)

        return y_pred_list


