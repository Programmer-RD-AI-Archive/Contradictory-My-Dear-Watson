import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import *
from torch.optim import *
from torchvision.models import *
from sklearn.model_selection import *
from sklearn.metrics import *
import wandb
import nltk
from nltk.stem.porter import *
PROJECT_NAME = "Contradictory, My Dear Watson"
np.random.seed(55)
stemmer = PorterStemmer()
device = 'cuda'


def tokenize(word):
    words = nltk.word_tokenize(word)
    return words


tokenize("Testing organizing")


def stem(word):
    word = stemmer.stem(word)
    return word


stem("Organic")


def converter(words,all_words):
    n_words = []
    for w in words:
        n_words.append(stem(w))
    np_eyes = np.zeros(len(all_words))
    for i,w in enumerate(all_words):
        if w in n_words:
            np_eyes[i] = 1.0
    return np_eyes


converter(['testing'],['test'])


data = pd.read_csv('./data/train.csv')


data



