import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from PIL import Image

# 配置参数（可调整超参数）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32  # 批处理大小（可调整：16/32/64）
epochs = 3  # 训练轮次（可调整：5/10/20）
lr = 0.001  # 学习率（可调整：0.01/0.001/0.0001）
data_dir = "D:\\python_project\\pythonProject\\test_03\\dc_2000"  # 数据集根目录
save_path = "cat_dog_best_model.pth"  # 模型保存路径

# 记录训练过程的指标（用于可视化）
train_losses = []
train_accs = []
val_losses = []
val_accs = []


# 1. 数据预处理
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试集与验证集预处理一致
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform


# 2. 加载数据集（新增测试集加载）
def load_data():
    train_transform, val_test_transform = get_transforms()

    # 加载训练集、验证集、测试集（假设数据集有test文件夹）
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),  # 新增测试集
        transform=val_test_transform
    )

    # 数据加载器（Windows系统建议num_workers=0）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"训练集数量: {len(train_dataset)}, 验证集数量: {len(val_dataset)}, 测试集数量: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_dataset.classes


# 3. 构建模型
def build_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    for param in list(model.parameters())[:-10]:  # 冻结部分层
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# 4. 训练循环（完善指标记录）
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, classes):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算并记录指标
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        train_losses.append(train_loss_avg)
        train_accs.append(train_acc)
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train: Loss={train_loss_avg:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss_avg:.4f}, Acc={val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"最佳模型已保存至 {save_path}（验证准确率：{best_val_acc:.2f}%）")

    print(f"\n训练完成，最佳验证准确率：{best_val_acc:.2f}%")
    return model


# 5. 模型评估（在测试集上评估）
def evaluate_model(model, test_loader, classes):
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    wrong_images = []  # 记录分类错误的图像

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 收集所有预测和标签（用于混淆矩阵）
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 记录错误样本
            wrong_idx = (predicted != labels).nonzero().squeeze()
            if wrong_idx.numel() > 0:
                if wrong_idx.dim() == 0:  # 单个错误样本
                    wrong_idx = [wrong_idx.item()]
                else:
                    wrong_idx = wrong_idx.tolist()
                wrong_images.extend([(images[i].cpu(), labels[i].cpu(), predicted[i].cpu()) for i in wrong_idx])

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"\n测试集准确率：{test_acc:.2f}%")

    # 打印分类报告（精确率、召回率、F1分数）
    print("\n分类报告：")
    print(classification_report(all_labels, all_preds, target_names=classes))

    return all_labels, all_preds, wrong_images, test_acc


# 6. 结果可视化
def visualize_results(classes, wrong_images):
    # 6.1 绘制学习曲线（损失）
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 6.2 绘制学习曲线（准确率）
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label="Train Acc")
    plt.plot(range(1, epochs + 1), val_accs, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.show()

    # 6.3 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 6.4 可视化错误分类的图像（最多显示8张）
    if len(wrong_images) > 0:
        plt.figure(figsize=(12, 6))
        show_num = min(8, len(wrong_images))
        for i in range(show_num):
            img, true_label, pred_label = wrong_images[i]
            # 反归一化图像以便显示
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            plt.subplot(2, 4, i + 1)
            plt.imshow(img)
            plt.title(f"True: {classes[true_label]}\nPred: {classes[pred_label]}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("wrong_predictions.png")
        plt.show()


# 7. 模型加载与推理
def load_model_and_predict(model_path, image_path, classes):
    # 加载模型
    model = build_model(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 推理
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

    result = classes[predicted.item()]
    print(f"\n推理结果：{result}（置信度：{confidence:.2f}%）")
    return result


# 主函数
if __name__ == "__main__":
    # 加载数据（包含测试集）
    train_loader, val_loader, test_loader, classes = load_data()

    # 初始化模型、损失函数、优化器
    model = build_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 可替换为SGD等优化器

    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, classes)

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(save_path))
    all_labels, all_preds, wrong_images, test_acc = evaluate_model(model, test_loader, classes)

    # 结果可视化
    visualize_results(classes, wrong_images)

    # 示例：加载模型进行单张图片推理（替换为你的测试图片路径）
    test_image_path = "test.jpg"  # 测试图片路径
    if os.path.exists(test_image_path):
        load_model_and_predict(save_path, test_image_path, classes)
    else:
        print(f"\n测试图片 {test_image_path} 不存在，跳过单张推理")