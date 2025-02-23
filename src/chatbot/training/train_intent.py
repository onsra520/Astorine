import sys
import os
import json5
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def configure_path():
    try:
        project_root = Path(__file__).resolve().parents[3]
    except NameError:
        project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    return project_root

project_root = configure_path()
from chatbot.utils.nlp_utils import tokenize, stem, bag_of_words
from chatbot.nlp.ChatModel import NeuralNetwork, ChatDataset

base_dir = os.path.join(project_root, "src", "chatbot")
paths = {
    "models": os.path.join(base_dir, "models"),
    "encoders": os.path.join(base_dir, "models", "encoders"),
    "scalers": os.path.join(base_dir, "models", "scalers"),
    "intents": os.path.join(base_dir, "intents", "intents.json"),
}

with open(paths["intents"], "r", encoding="utf-8") as f:
    intents = json5.load(f)

all_words, tags, xy = [], [], []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))

# stem và lower mỗi từ
ignore_words = ['?', '.', '!']
all_words = [stem(word) for word in all_words if word not in ignore_words]

# Xóa các từ trùng lặp và sắp xếp chúng
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Tạo dữ liệu huấn luyện
X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    # X: Túi từ cho mỗi câu mẫu (pattern_sentence).
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # Y: PyTorch CrossEntropyLoss chỉ cần class labels, không cần one-hot.
    label = tags.index(tag)
    Y_train.append(label)
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Hyper-parameters - Các siêu tham số
num_epochs = 500 # Số lần lặp qua toàn bộ dữ liệu huấn luyện
batch_size = 8 # Số mẫu dữ liệu được xử lý cùng một lúc trong mỗi lần cập nhật trọng số.
learning_rate = 0.001 # Tốc độ học, điều chỉnh độ lớn của bước cập nhật trọng số.
input_size = len(X_train[0]) # Kích thước đầu vào của mô hình, bằng số lượng từ trong bộ từ vựng (all_words).
hidden_size = 8 # Kích thước của lớp ẩn (hidden layer) trong mô hình.
output_size = len(tags) # Kích thước đầu ra của mô hình, bằng số lượng nhãn (tags).

dataset = ChatDataset(X_train, Y_train)
train_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Nếu có GPU thì sử dụng, nếu không thì sử dụng CPU
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình qua nhiều epoch.
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass - Lan truyền tiến là quá trình mô hình tính toán đầu ra (output) dựa trên dữ liệu đầu vào (input)
        outputs = model(words)
        loss = criterion(outputs, labels) # Tính toán loss giữa outputs và labels.

        # Backward and optimize - Lan truyền ngược và tối ưu hóa
        optimizer.zero_grad() # Xóa gradient từ trước đó, tránh việc gradient tích lũy.
        loss.backward() # thực hiện lan truyền ngược, tính toán gradient của loss đối với tất cả các tham số của mô hình bằng cách sử dụng quy tắc chuỗi (chain rule).
        optimizer.step() # cập nhật trọng số của mô hình dựa trên gradient đã tính toán.

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

torch.save(data, os.path.join(paths["models"], "chatbot.pth")) 
print(f'training complete. file saved to {paths["models"]}')
