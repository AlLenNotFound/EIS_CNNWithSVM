from collections import Counter
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from feature_extract import extract_all_curves
from model import CurveCNN, train_model, evaluate_model


class CurveDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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
#
# curve_idx = 10
# x_smooth = np.linspace(x_resampled[:, curve_idx].min(), x_resampled[:, curve_idx].max(), 500)
# y_smooth = fitted_curves[curve_idx](x_smooth)
# plt.scatter(x[:, curve_idx], y[:, curve_idx], label="original", color="red")
# plt.plot(x_smooth, y_smooth, label="Interpolation", color="blue")
# plt.legend()
# plt.title(f"curve {curve_idx} ")
# plt.show()

all_features = extract_all_curves(x_resampled, fitted_curves)
print(f"特征矩阵形状: {all_features.shape}")

all_features_tensor = torch.tensor(all_features, dtype=torch.float32)  # 转为 PyTorch 张量
labels_tensor = torch.tensor(labels, dtype=torch.long) - 1  # 标签转为整数张量
print(f"调整后标签范围: {labels_tensor.min()} 到 {labels_tensor.max()}")

X_train, X_test, y_train, y_test = train_test_split(all_features_tensor, labels_tensor, test_size=0.1, random_state=42)

# 创建 DataLoader
train_dataset = CurveDataset(X_train, y_train)
test_dataset = CurveDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(Counter(labels))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CurveCNN(num_features=30, num_classes=5).to(device)  # 假设有 5 个类别
criterion = nn.CrossEntropyLoss()  # 分类任务常用损失函数# 定义优化器，添加 weight_decay 参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)


train_model(model, train_loader, criterion, optimizer, num_epochs=60)
evaluate_model(model, test_loader)

train_loss = []
test_loss = []

from sklearn.metrics import confusion_matrix
import seaborn as sns

# 获取预测和真实标签
y_pred = []
y_true = []

model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features.to(device))
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

