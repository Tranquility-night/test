import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

# ----------------------------
# 配置参数（根据你的需求修改）
# ----------------------------
# 数据集路径（你的BSD300路径）
train_data_dir = r'D:\python_project\pythonProject\test_06\BSD300\train'  # 存放干净训练图像
# 模型保存路径
model_save_dir = './models'
# 训练参数
batch_size = 16  # 批次大小
epochs = 4  # 训练轮数
sigma = 25  # 噪声水平（高斯噪声标准差）
lr = 1e-3  # 学习率
image_size = 128  # 训练图像裁剪尺寸
channels = 1  # 图像通道数（3=彩色，1=灰度，根据你的数据修改）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

# 创建模型保存目录
os.makedirs(model_save_dir, exist_ok=True)


# ----------------------------
# 定义DnCNN模型
# ----------------------------
class DnCNN(nn.Module):
    def __init__(self, channels, num_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64  # 卷积核数量

        # 第一层：卷积+ReLU
        layers = [
            nn.Conv2d(channels, features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]

        # 中间层：卷积+BN+ReLU（共num_layers-2层）
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))  # 批量归一化加速训练
            layers.append(nn.ReLU(inplace=True))

        # 最后一层：卷积（输出噪声残差）
        layers.append(nn.Conv2d(features, channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 去噪逻辑：去噪图像 = 带噪图像 - 噪声残差（模型输出）
        residual = self.model(x)
        return x - residual


# ----------------------------
# 自定义训练数据集
# ----------------------------
class DenoisingDataset(Dataset):
    def __init__(self, data_dir, image_size, sigma, transform=None):
        # 获取所有图像路径（支持png/jpg/jpeg）
        self.image_paths = glob(os.path.join(data_dir, '*.[pj][np]g')) + glob(os.path.join(data_dir, '*.jpeg'))
        self.image_size = image_size  # 裁剪尺寸
        self.sigma = sigma  # 噪声水平
        self.transform = transform  # 数据转换

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_dir} 中未找到图像文件，请检查路径是否正确！")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB' if channels == 3 else 'L')  # 彩色/灰度

        # 数据增强：随机裁剪（固定尺寸）
        transform = transforms.Compose([
            transforms.RandomCrop(self.image_size),  # 随机裁剪
            transforms.ToTensor()  # 转换为Tensor，像素值归一化到[0,1]
        ])
        clean_img = transform(img)  # 干净图像

        # 生成带噪图像（干净图像 + 高斯噪声）
        noise = torch.randn_like(clean_img) * (self.sigma / 255.0)  # 噪声归一化到[0,1]范围
        noisy_img = clean_img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)  # 确保像素值在[0,1]内

        return noisy_img, clean_img  # 输入：带噪图像；标签：干净图像


# ----------------------------
# 训练流程
# ----------------------------
def train():
    # 加载数据集
    train_dataset = DenoisingDataset(
        data_dir=train_data_dir,
        image_size=image_size,
        sigma=sigma
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 打乱数据
        num_workers=4  # 多线程加载（根据CPU核心数调整）
    )
    print(f"训练集加载完成，共 {len(train_dataset)} 张图像，批次大小 {batch_size}")

    # 初始化模型、损失函数、优化器
    model = DnCNN(channels=channels).to(device)
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

    # 记录损失
    loss_history = []

    # 开始训练
    print(f"开始训练（设备：{device}）...")
    for epoch in range(epochs):
        model.train()  # 训练模式（启用BN和Dropout）
        epoch_loss = 0.0

        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            # 数据移至设备
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # 前向传播
            outputs = model(noisy_imgs)  # 模型输出去噪图像
            loss = criterion(outputs, clean_imgs)  # 计算损失

            # 反向传播+参数更新
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 累计损失
            epoch_loss += loss.item()
            # 打印批次信息
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] 完成，平均损失: {avg_loss:.6f}\n")

        # 学习率衰减
        scheduler.step()

        # 保存模型（每5轮保存一次，最后一轮必保存）
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            model_path = os.path.join(model_save_dir, f"dncnn_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存至：{model_path}")

    # 绘制损失曲线
    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.savefig('training_loss.png')
    plt.close()
    print("训练完成！损失曲线已保存为 training_loss.png")


if __name__ == "__main__":
    train()