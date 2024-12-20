import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# define model
class Vae(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Vae, self).__init__()
        # encode
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        # decode
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var


# loss
def vae_loss(recon_x, x, mu, log_var):
    # 交叉墒损失
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # kl散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


# dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# initialization model and optional
input_dim = 784  # MNIST图像大小 28x28
hidden_dim = 400
latent_dim = 20
model = Vae(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train
num_epochs = 10000
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = vae_loss(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')

# generate
with torch.no_grad():
    z = torch.randn(64, latent_dim)
    generated_samples = model.decoder(z).view(-1, 1, 28, 28)
