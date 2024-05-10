
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
import pickle
from transformers import AutoTokenizer, AutoModel

def split_label(data):
    ner_labels = []
    pos_labels = []
    for i in data:
        ner_labels.append(i[0])
        pos_labels.append(i[1])
    return torch.tensor(ner_labels), torch.tensor(pos_labels)



if __name__ == '__main__':
    with open('sample_data/input_encode.pkl', 'rb') as f:
        encode_inputs = pickle.load(f)
    
    with open('sample_data/encode_label.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('sample_data/pos_dict_i2w.pkl', 'rb') as f:
        pos_dict_i2w = pickle.load(f)
    with open('sample_data/ner_dict_i2w.pkl', 'rb') as f:
        ner_dict_i2w = pickle.load(f)
    # Khởi tạo mô hình
    input_size = 768  # Kích thước vector nhúng từ
    hidden_size = 100  # Số nút trong tầng ẩn
    out_pos_size = len(pos_dict_i2w)  # Số lớp POS
    out_ner_size = len(ner_dict_i2w)  # Số lớp NER
    inputW_size = 100
    model = Model(input_size, hidden_size, out_pos_size,out_ner_size, inputW_size)

    # Hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Dữ liệu đầu vào
    

    phoBERT = AutoModel.from_pretrained("vinai/phobert-base-v2")

    # Huấn luyện mô hình
    for epoch in range(5):
        for i in range(len(encode_inputs)):
            optimizer.zero_grad()
            inputs = phoBERT(torch.tensor([encode_inputs[i]])).last_hidden_state 
            outputs_pos, output_ner = model(inputs[0,1:-1])
            ner_labels, pos_labels = split_label(labels[i])
            loss = criterion(outputs_pos, pos_labels) + criterion(output_ner, ner_labels)
            loss.backward()
            optimizer.step()
            print("Done")
    

