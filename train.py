import torch
import torch.nn as nn
from model import Model
import pickle
from transformers import AutoTokenizer, AutoModel

def split_data(data):
    return data.split()

if __name__ == '__main__':
    input_size = 768
    hidden_size = 100
    out_pos_size = 21
    out_ner_size = 8
    inputW_size = 100
    outputW_size = 21

    model = Model(input_size, hidden_size, out_pos_size, out_ner_size, inputW_size, outputW_size)

    with open('sample_data/input_sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)
    with open('sample_data/input_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    

