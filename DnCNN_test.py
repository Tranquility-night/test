import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

# ----------------------------
# 配置参数（已确认channels=1为灰度图）
# ----------------------------
test_data_dir = r'D:\python_project\pythonProject\test_06\BSD300\test'
model_path = r'D:\python_project\pythonProject\test_06\models\dncnn_epoch4.pth'
save_dir = './denoised_results'
sigma = 25  # 与训练时一致
channels = 1  # 灰度图，固定为1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(save_dir, exist_ok=True)


# ----------------------------
# DnCNN模型（与训练时一致）
# ----------------------------
class DnCNN(nn.Module):
    def __init__(self, channels, num_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        layers = [
            nn.Conv2d(channels, features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.model(x)
        return x - residual


# ----------------------------
# 自定义测试数据集
# ----------------------------
class DenoisingTestDataset(Dataset):
    def __init__(self, data_dir, sigma, transform=None):
        self.image_paths = glob(os.path.join(data_dir, '*.[pj][np]g')) + glob(os.path.join(data_dir, '*.jpeg'))
        self.sigma = sigma
        self.transform = transform

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_dir} 中未找到图像文件，请检查路径！")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # 灰度图加载（与channels=1匹配）
        img = Image.open(img_path).convert('L')  # 'L'明确指定为灰度图

        # 转换为Tensor（保持灰度图维度）
        transform = transforms.Compose([
            transforms.ToTensor()  # 输出 shape: (1, H, W)
        ])
        clean_img = transform(img)

        # 生成带噪图像
        noise = torch.randn_like(clean_img) * (self.sigma / 255.0)
        noisy_img = clean_img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        return noisy_img, clean_img, img_name


# ----------------------------
# 测试流程（修复灰度图维度处理）
# ----------------------------
def test():
    test_dataset = DenoisingTestDataset(
        data_dir=test_data_dir,
        sigma=sigma
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    )
    print(f"测试集加载完成，共 {len(test_dataset)} 张图像")

    # 加载模型
    model = DnCNN(channels=channels).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型加载成功：{model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请检查路径！")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}")

    model.eval()

    with torch.no_grad():
        for idx, (noisy_imgs, clean_imgs, img_names) in enumerate(test_loader):
            noisy_imgs = noisy_imgs.to(device)
            denoised_imgs = model(noisy_imgs)

            # 灰度图维度处理（关键修复）
            # 灰度图shape为 (1, H, W)，挤压通道维度后为 (H, W)，无需permute
            noisy_img = noisy_imgs.cpu().squeeze().numpy()  # 移除batch和通道维度 -> (H, W)
            clean_img = clean_imgs.cpu().squeeze().numpy()
            denoised_img = denoised_imgs.cpu().squeeze().numpy()

            # 转换为[0,255]像素值
            noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
            clean_img = np.clip(clean_img * 255, 0, 255).astype(np.uint8)
            denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)

            # 保存结果（灰度图直接保存）
            img_name = img_names[0]
            denoised_save_path = os.path.join(save_dir, f"denoised_{img_name}")
            Image.fromarray(denoised_img).save(denoised_save_path)

            print(f"已处理 {idx + 1}/{len(test_dataset)}：{denoised_save_path}")

    print("测试完成！所有结果已保存至：", save_dir)


if __name__ == "__main__":
    test()