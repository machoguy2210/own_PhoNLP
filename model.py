import torch
import torch.nn as nn
from POS_layer import POS
from NER_layer import NER
from transformers import AutoModel,AutoTokenizer

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_pos_size, out_ner_size, inputW_size):
        super(Model, self).__init__()
        self.pos = POS(input_size, hidden_size, out_pos_size)
        self.ner = NER(input_size, hidden_size, out_ner_size, inputW_size, out_pos_size)

    def forward(self, x):
        p = self.pos(x)
        n = self.ner(x, p)
        return p, n

if __name__ == '__main__':
    input_size = 768
    hidden_size = 100
    out_pos_size = 5
    out_ner_size = 5
    inputW_size = 5

    model = Model(input_size, hidden_size, out_pos_size, out_ner_size, inputW_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    sentences = "Tôi là sinh_viên nghiên_cứu ."
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    encode_input = torch.tensor([tokenizer.encode(sentences)])
    phoBERT = AutoModel.from_pretrained("vinai/phobert-base-v2")
    inputs = phoBERT(encode_input).last_hidden_state
    outputs_pos, outputs_ner = model(inputs[0,1:-1])
    print(outputs_pos)
    print(outputs_ner)



