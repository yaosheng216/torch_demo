import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义一些超参数
batch_size = 64
num_epochs = 100
beta_start = 1e-4
beta_end = 0.02
# 扩散步长
num_timesteps = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 计算beta schedule
betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])

# 计算逆向过程所需的参数
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def q_sample(x_start, t, noise=None):
    if noise is None:
        # 生成一个和x_start具有相同shape的高斯分布tensor
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    # 在tensor a中按照t最后一维的形状手机元素为-1的新tensor
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianUNet(nn.Module):
    def __init__(self):
        super(GaussianUNet, self).__init__()
        # 下采样路径
        self.down1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 上采样路径
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 下采样
        x1 = self.relu(self.down1(x))
        x2 = self.pool(x1)
        x2 = self.relu(self.down2(x2))
        x2 = self.pool(x2)

        # 上采样
        x = self.up1(x2)
        x = self.relu(self.up2(x))
        return x


model = GaussianUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        # 整数tensor，代表当前要扩散的步长
        t = torch.randint(0, num_timesteps, (images.size(0),), device=device).long()
        # 具有标准高斯分布的tensor（噪声）
        noise = torch.randn_like(images)
        # 根据原图，步长和噪点图生成扩散图
        x_noise = q_sample(images, t, noise=noise)

        predicted_noise = model(x_noise)
        loss = nn.functional.mse_loss(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{len(train_loader)}], loss: {loss.item():.4f}')

print('训练完成！')
