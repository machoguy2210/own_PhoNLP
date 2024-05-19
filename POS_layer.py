import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
import pickle


# Mô hình FNN
class POS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(POS, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    with open('sample_data/input_encode.pkl', 'rb') as f:
        encode_inputs = pickle.load(f)
    
    with open('sample_data/pos_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('sample_data/pos_dict_w2i.pkl', 'rb') as f:
        pos_dict_w2i = pickle.load(f)
    # Khởi tạo mô hình
    input_size = 768  # Kích thước vector nhúng từ
    hidden_size = 100  # Số nút trong tầng ẩn
    output_size = len(pos_dict_w2i)  # Số lớp POS
    model = POS(input_size, hidden_size, output_size)

    # Hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Dữ liệu đầu vào
    

    phoBERT = AutoModel.from_pretrained("vinai/phobert-base-v2")

    # Huấn luyện mô hình
    for epoch in range(1):
        for i in range(len(encode_inputs)):
            optimizer.zero_grad()
            inputs = phoBERT(torch.tensor([encode_inputs[i]])).last_hidden_state 
            outputs = model(inputs[0,1:-1])
            target = torch.tensor(list(map(lambda x: pos_dict_w2i[x],labels[i])))
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            print("Done")

    
