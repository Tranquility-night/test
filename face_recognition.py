import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm import tqdm

# ----------------------------
# 配置参数（根据需求修改）
# ----------------------------
DATA_DIR = "D:\\python_project\\pythonProject\\test_05"  # 替换为你的数据集根目录
BATCH_SIZE = 32  # 批处理大小
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 4  # 训练轮数
VAL_SPLIT = 0.2  # 从训练集中划分20%作为验证集
MODEL_SAVE_PATH = "trained_model.pth"  # 模型保存路径


# ----------------------------
# 1. 初始化与数据准备（无独立验证集时从训练集拆分）
# ----------------------------
def setup_environment():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


def load_data(data_dir, batch_size, val_split):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集和测试集
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    # 从训练集中拆分验证集
    val_size = int(val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定拆分方式
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(
        f"数据集加载完成 - 训练集: {len(train_dataset)} 样本, 验证集(拆分): {len(val_dataset)} 样本, 测试集: {len(test_dataset)} 样本")
    return train_loader, val_loader, test_loader, full_train_dataset.classes  # 类别名从原始训练集获取


# ----------------------------
# 2. 模型定义（不变）
# ----------------------------
def build_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


# ----------------------------
# 3. 模型训练（不变）
# ----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (训练)"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_avg_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history["train_loss"].append(train_avg_loss)
        history["train_acc"].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (验证)"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history["val_loss"].append(val_avg_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"训练集 - 损失: {train_avg_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证集 - 损失: {val_avg_loss:.4f}, 准确率: {val_acc:.4f}\n")

    return model, history


# ----------------------------
# 4. 模型评估（不变）
# ----------------------------
def evaluate_model(model, test_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试集评估"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"\n测试集准确率: {test_acc:.4f}")

    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return all_labels, all_preds


# ----------------------------
# 5. 结果可视化（不变）
# ----------------------------
def visualize_results(history, true_labels, pred_labels, class_names, model, test_loader, device):
    # 1. 损失和准确率曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="训练损失")
    plt.plot(history["val_loss"], label="验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.title("损失曲线")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="训练准确率")
    plt.plot(history["val_acc"], label="验证准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.title("准确率曲线")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_acc_curve.png")
    plt.show()
    print("损失/准确率曲线已保存为 loss_acc_curve.png")

    # 2. 混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("混淆矩阵已保存为 confusion_matrix.png")

    # 3. 错误分类样本
    model.eval()
    misclassified = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(inputs)):
                if preds[i] != labels[i]:
                    misclassified.append({
                        "image": inputs[i].cpu().permute(1, 2, 0),
                        "true": class_names[labels[i]],
                        "pred": class_names[preds[i]]
                    })
                    if len(misclassified) >= 5:
                        break
            if len(misclassified) >= 5:
                break

    plt.figure(figsize=(15, 3))
    for i, item in enumerate(misclassified):
        plt.subplot(1, 5, i + 1)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = item["image"] * std + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"真实: {item['true']}\n预测: {item['pred']}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_samples.png")
    plt.show()
    print("错误分类样本已保存为 misclassified_samples.png")


# ----------------------------
# 6. 模型保存与加载（不变）
# ----------------------------
def save_model(model, optimizer, class_names, save_path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names
    }, save_path)
    print(f"模型已保存至 {save_path}")


def load_and_infer(save_path, num_classes, test_dataset, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint["class_names"]
    model = model.to(device)
    model.eval()
    print(f"模型已从 {save_path} 加载")

    # 随机选一个测试样本推理
    idx = np.random.randint(0, len(test_dataset))
    sample_img, sample_label = test_dataset[idx]
    with torch.no_grad():
        img_tensor = sample_img.unsqueeze(0).to(device)
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    print(f"\n推理示例 - 真实类别: {class_names[sample_label]}, 预测类别: {class_names[pred.item()]}")


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 1. 环境设置
    device = setup_environment()

    # 2. 数据加载（含验证集拆分）
    train_loader, val_loader, test_loader, class_names = load_data(DATA_DIR, BATCH_SIZE, VAL_SPLIT)
    num_classes = len(class_names)
    print(f"类别列表: {class_names}")

    # 3. 模型构建
    model, criterion, optimizer = build_model(num_classes, device)

    # 4. 模型训练
    print("\n开始训练...")
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS
    )

    # 5. 模型评估
    print("\n开始测试集评估...")
    true_labels, pred_labels = evaluate_model(trained_model, test_loader, class_names, device)

    # 6. 结果可视化
    print("\n生成可视化结果...")
    visualize_results(history, true_labels, pred_labels, class_names, trained_model, test_loader, device)

    # 7. 模型保存与加载示例
    save_model(trained_model, optimizer, class_names, MODEL_SAVE_PATH)
    load_and_infer(MODEL_SAVE_PATH, num_classes, test_loader.dataset, device)


if __name__ == "__main__":
    main()