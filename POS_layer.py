import torch
import torch.nn as nn
import torch.optim as optim
import transformers
# Dữ liệu ví dụ (vector nhúng từ và nhãn POS tương ứng)
# Đây chỉ là ví dụ, bạn cần thay thế bằng dữ liệu thực tế của bạn


# Mô hình FNN
class POS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(POS, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Khởi tạo mô hình
input_size = 300  # Kích thước vector nhúng từ
hidden_size = 100  # Số nút trong tầng ẩn
output_size = 5  # Số lớp POS
model = POS(input_size, hidden_size, output_size)

# Hàm mất mát và tối ưu hóa

optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(300)


output = torch.tensor([0.2813, 0.1676, 0.1858, 0.1886, 0.1767])
target = torch.tensor(1)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(loss)

'''# Huấn luyện mô hình
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()'''

# Dự đoán1
'''with torch.no_grad():
    test_word = torch.randn(1, 300)  # Từ mới cần dự đoán
    predicted_probs = model(test_word)
    _, predicted_label = torch.max(predicted_probs, 1)
    print(f"Dự đoán POS: Lớp {predicted_label.item()}")'''
