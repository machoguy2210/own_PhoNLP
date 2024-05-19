import torch
import torch.nn as nn

class NER(nn.Module):
    def __init__(self, input_size,hidden_size, ouput_size, inputW_size,outputW_size):
        super(NER, self).__init__()
        self.fc1 = nn.Linear(input_size + inputW_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, ouput_size)
        self.W = nn.Parameter(torch.rand(inputW_size, outputW_size))

    def forward(self, e,p):
        t = []
        for vector in p:
            _t = torch.matmul(self.W, vector)
            t.append(_t)
        t = torch.stack(t)
        v = torch.cat((e, t),dim= 1)
        h = torch.relu(self.fc1(v))
        h = self.fc2(h)
        return h

if __name__ == '__main__':
    input_size = 300
    hidden_size = 100
    output_size = 5
    inputW_size = 5
    outputW_size = 5
    
    model = NER(input_size, hidden_size, output_size, inputW_size, outputW_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    sample_input = torch.randn(1, input_size)



        
        

