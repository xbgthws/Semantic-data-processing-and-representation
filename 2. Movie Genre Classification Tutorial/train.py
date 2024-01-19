# 读取电影评论文件movie.csv，以电影的名字和评论作为输入，电影的类型作为输出来实现对电影类型的预测。编写一个LSTM模型，对电影类型进行预测。

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# 读取数据
df = pd.read_csv('movie.csv', encoding='utf-8')
print(df.head())

# 读取电影名和评论
movie_name = df['movie name'].values
comment = df['comment'].values

# 读取电影类型
movie_type = df['type'].values


# 数据预处理
def preprocess(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去除数字
    text = re.sub(r'\d+', '', text)
    return text

# 将电影名和评论进行预处理
movie_name = [preprocess(x) for x in movie_name]
comment = [preprocess(x) for x in comment]
# 合并电影名和评论
text = [x + y for x, y in zip(movie_name, comment)]

# 将电影类型转换为数字
type2id = dict(zip(set(movie_type), range(len(set(movie_type)))))
print(type2id)

# 构建词典
word2id = {}
for text in movie_name + comment:
    for word in text.split():
        if word not in word2id:
            word2id[word] = len(word2id)

# 采用word2vec的方式将文本转换为数字
from gensim.models import Word2Vec


# 训练词向量
def train_word2vec(text, save_path):
    model = Word2Vec(text, size=100, window=5, min_count=1, workers=4)
    model.save(save_path)
    return model


# 划分训练集和测试集
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(text, movie_type,  test_size=0.2, random_state=1)

# 转换为Tensor
train_x = torch.LongTensor(train_x)
train_y = torch.LongTensor(train_y)
test_x = torch.LongTensor(test_x)
test_y = torch.LongTensor(test_y)

# 构建数据集
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

# 定义模型参数
vocab_size = len(word2id)
embedding_dim = 100
hidden_dim = 128
num_layers = 1
num_classes = len(type2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
best_acc = 0.0
train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []
print('start training...')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += (outputs.argmax(1) == labels).sum().item()
        train_loss += loss.item()
    train_acc_list.append(train_acc / len(train_dataset))
    train_loss_list.append(train_loss / len(train_dataset))

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_acc += (outputs.argmax(1) == labels).sum().item()
            test_loss += loss.item()
    test_acc_list.append(test_acc / len(test_dataset))
    test_loss_list.append(test_loss / len(test_dataset))
    # 保存最好的模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best.pth')

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch + 1, num_epochs, train_loss / len(train_dataset), train_acc / len(train_dataset),
                  test_loss / len(test_dataset), test_acc / len(test_dataset)))

print('Finished Training')







