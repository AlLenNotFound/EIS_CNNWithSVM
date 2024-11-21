import torch
import torch.nn as nn
import torch.nn.functional as F


class CurveCNN(nn.Module):
    def __init__(self, num_features=20, num_classes=5):
        super(CurveCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 卷积层
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * num_features, 64)  # 全连接层
        self.fc2 = nn.Linear(64, num_classes)  # 输出分类层
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(F.relu(self.fc1(x)))  # 在全连接层后添加 Dropout
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()  # 设置为训练模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            # 将数据移动到设备 (如果有 GPU)
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():  # 评估时不需要计算梯度
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
