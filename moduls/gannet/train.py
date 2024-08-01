import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models import DiscriminatorNet, GeneratorNet
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from dataset import ImageDataset
import itertools
import tensorboardX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 1
size = 256
lr = 0.0002
n_epoch = 200
epoch = 0
decay_epoch = 100

# networks
netG_A2B = GeneratorNet().to(device)
netG_B2A = GeneratorNet().to(device)
netD_A = DiscriminatorNet().to(device)
netD_B = DiscriminatorNet().to(device)

# loss
loss_GAN = torch.nn.MSELoss()
loss_cycle = torch.nn.L1Loss()
loss_identity = torch.nn.L1Loss()

# optimizer & LR
opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),
                                         netG_B2A.parameters()), lr=lr, betas=(0.5, 0.9999))
opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.9999))
opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.9999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(n_epoch,  epoch, decay_epoch).step)


data_root = "datasets/apple2orange"
input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
label_real = torch.ones([1], dtype=torch.float, requires_grad=False).to(device)
label_fake = torch.zeros([1], dtype=torch.float, requires_grad=False).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

log_path = "logs"
writer_log = tensorboardX.SummaryWriter(log_path)

# 数据增强
transforms_ = [
    transforms.Resize(int(256 * 1.12), Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = DataLoader(ImageDataset(data_root, transforms_),
                        batch_size=batchsize, shuffle=True, num_workers=8)
step = 0
for epoch in range(n_epoch):
    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

        opt_G.zero_grad()

        same_B = netG_A2B(real_B)
        loss_identity_B = loss_identity(same_B, real_B) * 5.0

        same_A = netG_B2A(real_A)
        loss_identity_A = loss_identity(same_A, real_A) * 5.0

        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = loss_GAN(pred_fake, label_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = loss_GAN(pred_fake, label_real)

        # cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0
        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0

        loss_G = loss_identity_A + loss_identity_B + \
                 loss_GAN_A2B + loss_GAN_B2A + \
                 loss_cycle_ABA + loss_cycle_BAB

        loss_G.backward()
        opt_G.step()
        opt_DA.zero_grad()
        pred_real = netD_A(real_A)
        loss_D_real = loss_GAN(pred_real, label_real)

        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = loss_GAN(pred_fake, label_fake)

        # total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        opt_DA.step()

        opt_DB.zero_grad()
        pred_real = netD_B(real_B)
        loss_D_real = loss_GAN(pred_real, label_real)

        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = loss_GAN(pred_fake, label_fake)

        # total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        opt_DB.step()

        print("loss_G:{}, loss_G_identity:{}, loss_G_GAN:{}, "
              "loss_G_cycle:{}, loss_D_A:{}, loss_D_B:{}".format(
            loss_G, loss_identity_A + loss_identity_A,
                  loss_GAN_A2B + loss_GAN_B2A,
                  loss_cycle_BAB + loss_cycle_ABA,
                  loss_D_A, loss_D_B
        ))

        writer_log.add_scalar("loss_G", loss_G, global_step=step + 1)
        writer_log.add_scalar("loss_G_identity", loss_identity_A + loss_identity_A, global_step=step + 1)
        writer_log.add_scalar("loss_G_GAN", loss_GAN_A2B + loss_GAN_B2A, global_step=step + 1)
        writer_log.add_scalar("loss_G_cycle", loss_cycle_BAB + loss_cycle_ABA, global_step=step + 1)
        writer_log.add_scalar("loss_D_A", loss_D_A, global_step=step + 1)
        writer_log.add_scalar("loss_D_B", loss_D_B, global_step=step + 1)

        step += 1

    lr_scheduler_DA.step()
    lr_scheduler_DB.step()
    lr_scheduler_G.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'models/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'models/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'models/netD_A.pth')
    torch.save(netD_B.state_dict(), 'models/netD_B.pth')
