# Homework #1 Named Entity Recognition

This is the first homework of the NLP 2022 course at Sapienza University of Rome, taught by [**Roberto Navigli**](http://www.diag.uniroma1.it/~navigli/).

Check the [Report](https://github.com/florin-git/Named-Entity-Recognition/blob/main/report.pdf) and the Slide Presentation for more information.

The best model reaches an **F1 score of 74.0%** on the secret test set of the course.

## Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project

### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding.
For those who might be unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```bash
conda create -n nlp2022-hw1 python=3.7
conda activate nlp2022-hw1
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp2022-hw1
bash test.sh data/dev.tsv
```

Actually, you can replace *data/dev.tsv* to point to a different file, as far as the target file has the same format.

## Reproduce using checkpoints
You can download the checkpoints of the models I described in the paper from this [Google Drive link](https://drive.google.com/file/d/10cw9j-4NNjX8a76ZmMcyEzje02F4Ct3y/view?usp=sharing).

