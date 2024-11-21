import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from feature_extract import extract_all_curves
import pandas as pd


def resample_data(x, y, num_points=100):
    x_new = np.linspace(x.min(), x.max(), num_points)  # 生成等间距的 x_new
    y_new = np.interp(x_new, x, y)  # 对 y 进行线性插值
    return x_new, y_new


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)  # 均值
        self.fc2_logvar = nn.Linear(128, latent_dim)  # 方差

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, logvar, z


class CNNClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = self.fc2(h)
        return out


class VAE_CNN(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(VAE_CNN, self).__init__()
        self.vae = VAE(input_dim, latent_dim)
        self.cnn = CNNClassifier(latent_dim, num_classes)

    def forward(self, x):
        x_reconstructed, mean, logvar, z = self.vae(x)
        class_logits = self.cnn(z)
        return x_reconstructed, mean, logvar, class_logits


def vae_loss_function(x, x_reconstructed, mean, logvar):
    # 重构损失
    reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
    # KL 散度
    kl_divergence = -0.5 * torch.sum(1 + torch.clamp(logvar, -10, 10) - mean.pow(2) - logvar.exp())
    return reconstruction_loss, kl_divergence


data = pd.read_csv('C:\\Users\\李艺博\\Desktop\\data.csv', header=None)
labels = data.iloc[2, 1:].values
labels = labels.astype(int)
data_values = data.iloc[3:, 1:].astype(float).values
x = data_values[:80, :]
y = data_values[80:160, :]

num_resample_points = 100  # 重采样点数
x_resampled = np.zeros((num_resample_points, x.shape[1]))
y_resampled = np.zeros((num_resample_points, y.shape[1]))
for i in range(x.shape[1]):
    sort_idx = np.argsort(x[:, i])  # 确保 x 是递增的
    x_sorted = x[sort_idx, i]
    y_sorted = y[sort_idx, i]
    x_resampled[:, i], y_resampled[:, i] = resample_data(x_sorted, y_sorted, num_points=num_resample_points)

fitted_curves = []
for i in range(x_resampled.shape[1]):
    cs = CubicSpline(x_resampled[:, i], y_resampled[:, i], bc_type='natural')  # 使用 natural 边界条件
    fitted_curves.append(cs)

all_features = extract_all_curves(x_resampled, fitted_curves)
scaler = MinMaxScaler()
all_features = scaler.fit_transform(all_features)
# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 数据集划分
train_dataset = CustomDataset(all_features[:450], labels[:450])
test_dataset = CustomDataset(all_features[450:], labels[450:])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型和优化器
input_dim = 30  # 输入特征维度
latent_dim = 20  # 潜在空间维度 10
num_classes = 5  # 分类类别数

model = VAE_CNN(input_dim, latent_dim, num_classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


model.vae.apply(init_weights)
# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, labels in train_loader:
        x, labels = x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 前向传播
        x_reconstructed, mean, logvar, class_logits = model(x)

        # 计算损失
        reconstruction_loss, kl_divergence = vae_loss_function(x, x_reconstructed, mean, logvar)
        classification_loss = F.cross_entropy(class_logits, labels)
        loss = reconstruction_loss + 0.0001 * kl_divergence + 2 * classification_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vae.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, labels in test_loader:
        x, labels = x.to(torch.device('cpu')), labels.to(torch.device('cpu'))
        _, _, _, class_logits = model(x)
        _, predicted = torch.max(class_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

