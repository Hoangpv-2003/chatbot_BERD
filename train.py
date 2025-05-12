import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem  # Chắc chắn đã cập nhật tokenizers cho tiếng Việt
from model import NeuralNet

# Mở file JSON chứa intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Duyệt qua từng câu trong intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # Thêm tag vào danh sách
    for pattern in intent['patterns']:
        # Tokenize từng từ trong câu
        w = tokenize(pattern)
        all_words.extend(w)  # Thêm các từ đã tokenize vào danh sách all_words
        xy.append((w, tag))  # Thêm câu và tag vào danh sách xy

# Stem và chuyển tất cả các từ thành chữ thường
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Loại bỏ các từ trùng lặp và sắp xếp lại
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Tạo dữ liệu huấn luyện
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words cho mỗi câu pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss cần label class, không cần one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Các tham số mô hình
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])  # Số từ trong câu
hidden_size = 8  # Số lượng node trong hidden layer
output_size = len(tags)  # Số lượng tags (classes)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Tạo DataLoader để batch dữ liệu
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward và tối ưu hóa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# In kết quả cuối cùng
print(f'final loss: {loss.item():.4f}')

# Lưu mô hình vào file
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
