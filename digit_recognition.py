import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. 数据加载与预处理（不变）
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ----------------------------
# 2. 模型定义（不变）
# ----------------------------
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ----------------------------
# 3. 模型训练（不变）
# ----------------------------
def train_model():
    model = DigitRecognizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 5  # 训练5轮
    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f'Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%')

    return model, train_losses, train_accs


# ----------------------------
# 4. 模型评估（修复错误样本索引处理）
# ----------------------------
def evaluate_model(model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    misclassified = []  # 记录错误样本

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 修复：处理错误样本索引的维度问题
            if len(misclassified) < 5:
                # 找到错误样本的索引（保持维度为1D）
                wrong_mask = (predicted != labels)
                wrong_idx = torch.nonzero(wrong_mask, as_tuple=False).squeeze(dim=1)  # 确保是1D张量

                # 如果有错误样本，添加到列表
                if wrong_idx.numel() > 0:  # 检查是否有元素
                    # 取第一个错误样本
                    idx = wrong_idx[0].item()  # 转为标量索引
                    misclassified.append({
                        'image': inputs[idx].cpu().squeeze(),
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item()
                    })

    test_acc = 100 * correct / total
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    return test_acc, misclassified


# ----------------------------
# 5. 结果可视化（不变）
# ----------------------------
def visualize_results(train_losses, train_accs, misclassified):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 6), train_losses, 'b-')
    plt.title('Training Loss (5 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 6), train_accs, 'r-')
    plt.title('Training Accuracy (5 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.tight_layout()
    plt.show()

    if misclassified:
        plt.figure(figsize=(10, 4))
        for i, item in enumerate(misclassified):
            plt.subplot(1, len(misclassified), i+1)
            plt.imshow(item['image'], cmap='gray')
            plt.title(f'True: {item["true"]}\nPred: {item["pred"]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()


# ----------------------------
# 6. 模型保存与加载（不变）
# ----------------------------
def save_model(model, path='digit_model_5epochs.pth'):
    torch.save(model.state_dict(), path)
    print(f'\n模型已保存至 {path}')


def load_and_infer(image_path, model_path='digit_model_5epochs.pth'):
    model = DigitRecognizer()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    from PIL import Image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    print(f'推理结果：{predicted.item()}')
    return predicted.item()


# ----------------------------
# 主函数执行
# ----------------------------
if __name__ == '__main__':
    model, train_losses, train_accs = train_model()
    test_acc, misclassified = evaluate_model(model)
    visualize_results(train_losses, train_accs, misclassified)
    save_model(model)
    torch.save(model.state_dict(), 'test_01/mnist_model_params.pth')