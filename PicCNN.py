import os
from collections import Counter
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from feature_extract import extract_all_curves

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNN(nn.Module):
    def __init__(self, image_channels, image_size, num_features, num_classes):
        super(HybridCNN, self).__init__()
        # CNN 部分（处理图像）
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_out_size = (image_size // 2) * (image_size // 2) * 32  # 池化后大小
        self.dropout = nn.Dropout(p=0.5)

        # FC 部分（处理一维特征 + CNN 输出）
        self.fc1 = nn.Linear(self.conv_out_size + num_features, 128)  # 拼接特征
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, image, features):
        # 图像部分
        x = F.relu(self.conv1(image))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)

        # 拼接图像特征和一维特征
        combined = torch.cat((x, features), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x


class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, images, features, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # 图像数据
        self.features = torch.tensor(features, dtype=torch.float32)  # 一维特征
        self.labels = torch.tensor(labels, dtype=torch.long)  # 标签

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.features[idx], self.labels[idx]


def resample_data(x, y, num_points=100):
    x_new = np.linspace(x.min(), x.max(), num_points)  # 生成等间距的 x_new
    y_new = np.interp(x_new, x, y)  # 对 y 进行线性插值
    return x_new, y_new


data = pd.read_csv('C:\\Users\\李艺博\\Desktop\\data.csv', header=None)
labels = data.iloc[2, 1:].values
labels = labels.astype(int)
data_values = data.iloc[3:, 1:].astype(float).values
x = data_values[:80, :]
y = data_values[80:160, :]
print(f"数据形状: {x.shape, y.shape}, 标签数量: {len(labels)}")

num_resample_points = 128  # 重采样点数
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

# 数据加载
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# 加载图像数据
image_dir = "D:\\pythonProject\\img"
image_files = sorted(os.listdir(image_dir))
image_data = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    with Image.open(image_path) as img:
        img = img.convert("L")  # 转为灰度模式
        img = img.resize((128, 128))  # 调整大小为 128x128
        img_array = np.array(img)  # 转为 NumPy 数组
        image_data.append(img_array)
image_data = np.array(image_data)  # (N, H, W)
print(f"图片数组形状: {image_data.shape}")  # (图片数量, 高度, 宽度)

indices = np.arange(len(labels))
np.random.shuffle(indices)
# 按随机索引打乱数据
images = image_data[indices]
features = all_features[indices]
labels = labels[indices]

train_dataset = HybridDataset(image_data[:400], all_features[:400], labels[:400]-1)
test_dataset = HybridDataset(image_data[400:], all_features[400:], labels[400:]-1)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNN(image_channels=1, image_size=128, num_features=30, num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# class_counts = Counter(labels)  # labels 是所有数据的标签
# print("Class counts:", class_counts)
# # 计算权重（类别样本数量越少，权重越高）
# class_weights = torch.tensor([1.0 / class_counts[i] for i in range(1, len(class_counts) + 1)]).to(device)
# # 使用加权的交叉熵损失
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
Epoch = 10
for epoch in range(Epoch):
    model.train()
    total_loss = 0
    for images, features, labels in train_loader:
        images, features, labels = images.to(device), features.to(device), labels.to(device)
        # 前向传播
        outputs = model(images, features)
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{Epoch}], Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    for images, features, labels in test_loader:
        images, features, labels = images.to(device), features.to(device), labels.to(device)

        # 前向传播
        outputs = model(images, features)
        _, predicted = torch.max(outputs, 1)  # 获取预测类别

        print(f"Outputs: {outputs}")  # 输出 logits
        print(f"Predicted: {predicted}")  # 输出预测类别
        print(f"Labels: {labels}")  # 输出真实标签
        break
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, features, labels in test_loader:
        images, features, labels = images.to(device), features.to(device), labels.to(device)

        # 前向传播
        outputs = model(images, features)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# output_dir = "D:\\pythonProject\\img"
# os.makedirs(output_dir, exist_ok=True)
#
# # 遍历每条曲线并保存
# for curve_idx in range(x_resampled.shape[1]):
#     # 获取平滑曲线
#     x_smooth = np.linspace(x_resampled[:, curve_idx].min(), x_resampled[:, curve_idx].max(), 500)
#     y_smooth = fitted_curves[curve_idx](x_smooth)
#
#     # 绘制原始点和拟合曲线
#     plt.figure(figsize=(6, 4))
#     plt.scatter(x[:, curve_idx], y[:, curve_idx], color="red")
#     plt.plot(x_smooth, y_smooth, color="blue")
#     plt.axis('off')
#
#     # 保存图片
#     output_path = os.path.join(output_dir, f"curve_{curve_idx}.png")
#     plt.savefig(output_path, dpi=64)  # 保存为高分辨率图片
#     plt.close()  # 关闭当前图，释放内存
#
#     print(f"Curve {curve_idx} saved to {output_path}")