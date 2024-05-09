import torch
import torch.nn as nn
from POS_layer import POS
from NER_layer import NER

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_pos_size, out_ner_size, inputW_size, outputW_size):
        super(Model, self).__init__()
        self.pos = POS(input_size, hidden_size, out_pos_size)
        self.ner = NER(input_size, hidden_size, out_ner_size, inputW_size, outputW_size)

    def forward(self, x):
        pass



