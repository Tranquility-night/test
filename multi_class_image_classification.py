import tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 3
lr = 0.001
data_dir = "D:\\python_project\\pythonProject\\test_04"  # 数据集路径
save_path = "cifar10_best_model.pth"
writer = SummaryWriter("runs/cifar10_experiment")  # TensorBoard日志路径

# 数据预处理
transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集并划分训练集、验证集、测试集
dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 类别标签
classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

# 构建模型（ResNet18 迁移学习）
def build_model(num_classes=10):
    model = resnet18(pretrained=False)  # CIFAR-10 无需预训练
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_acc = 0.0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss_avg, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss_avg, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"模型已更新并保存至 {save_path}")

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    return model

# 模型评估
def evaluate_model(model, test_loader, classes):
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    wrong_images = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            wrong_idx = (predicted != labels).nonzero().squeeze()
            if wrong_idx.numel() > 0:
                if wrong_idx.dim() == 0:
                    wrong_idx = [wrong_idx.item()]
                else:
                    wrong_idx = wrong_idx.tolist()
                wrong_images.extend([(images[i].cpu(), labels[i].cpu(), predicted[i].cpu()) for i in wrong_idx])

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"\n测试集准确率: {test_acc:.2f}%")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    return all_labels, all_preds, wrong_images, test_acc

# 结果可视化
def visualize_results(classes, all_labels, all_preds, wrong_images):
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 错误分类图像可视化
    if len(wrong_images) > 0:
        plt.figure(figsize=(12, 8))
        show_num = min(12, len(wrong_images))
        for i in range(show_num):
            img, true_label, pred_label = wrong_images[i]
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            plt.subplot(3, 4, i+1)
            plt.imshow(img)
            plt.title(f"真实: {classes[true_label]}\n预测: {classes[pred_label]}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("wrong_predictions.png")
        plt.show()

# 模型加载与推理
def load_model_and_predict(model_path, image_index, test_dataset, classes):
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image, label = test_dataset[image_index]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item() * 100

    print(f"\n真实类别: {classes[label]}")
    print(f"预测类别: {classes[predicted.item()]}，置信度: {confidence:.2f}%")

    # 显示图像
    img = test_dataset[image_index][0].numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"真实: {classes[label]}\n预测: {classes[predicted.item()]}")
    plt.axis("off")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 构建模型
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

    # 加载最佳模型并评估
    model.load_state_dict(torch.load(save_path))
    all_labels, all_preds, wrong_images, test_acc = evaluate_model(model, test_loader, classes)

    # 结果可视化
    visualize_results(classes, all_labels, all_preds, wrong_images)

    # 示例：加载模型进行单张图片推理
    load_model_and_predict(save_path, 100, test_dataset, classes)

    # 关闭TensorBoard写入器
    writer.close()