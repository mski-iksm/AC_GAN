from torchvision.utils import save_image, make_grid
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(110, 384)
        deconv_params = {"kernel_size": (5, 5),
                         "stride": (2, 2),
                         "padding": (2, 2),
                         "output_padding": 1}
        self.deconv_1 = nn.ConvTranspose2d(
            in_channels=24, out_channels=192, **deconv_params)
        self.deconv_2 = nn.ConvTranspose2d(
            in_channels=192, out_channels=96, **deconv_params)
        self.deconv_3 = nn.ConvTranspose2d(
            in_channels=96, out_channels=3, **deconv_params)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        initialize_weights(self)

    def forward(self, inputs):
        x = inputs.view(-1, 110)
        x = self.linear_1(x)
        x = self.relu(x)
        x = x.view(-1, 24, 4, 4)
        x = self.deconv_1(x)
        x = self.relu(x)
        x = self.deconv_2(x)
        x = self.relu(x)
        x = self.deconv_3(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        conv_params_2 = {"padding": 1, "kernel_size": 3, "stride": 2}
        conv_params_1 = {"padding": 1, "kernel_size": 3, "stride": 1}
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=16, **conv_params_2)
        self.conv_2 = nn.Conv2d(
            in_channels=16, out_channels=32, **conv_params_1)
        self.conv_3 = nn.Conv2d(
            in_channels=32, out_channels=64, **conv_params_2)
        self.conv_4 = nn.Conv2d(
            in_channels=64, out_channels=128, **conv_params_1)
        self.conv_5 = nn.Conv2d(
            in_channels=128, out_channels=256, **conv_params_2)
        self.conv_6 = nn.Conv2d(
            in_channels=256, out_channels=512, **conv_params_1)
        self.linear_1 = nn.Linear(4*4*512, 11)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        initialize_weights(self)

    def forward(self, inputs):
        x = inputs.view(-1, 3, 32, 32)
        x = self.conv_1(x)
        x = self.leakyrelu(x)
        x = self.conv_2(x)
        x = self.leakyrelu(x)
        x = self.conv_3(x)
        x = self.leakyrelu(x)
        x = self.conv_4(x)
        x = self.leakyrelu(x)
        x = self.conv_5(x)
        x = self.leakyrelu(x)
        x = self.conv_6(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 4*4*512)
        x = self.linear_1(x)
        output_label = self.sigmoid(x[:, 0]).view(-1, 1)
        output_class = self.softmax(x[:, 1:])
        return output_label, output_class


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def train(g_model, d_model, b_criterion, c_criterion, d_optimizer, g_optimizer, data_loader,
          batch_size=32):
    """ single epoch
    """
    g_model.train()
    d_model.train()

    y_real = torch.ones(batch_size, 1)
    y_fake = torch.zeros(batch_size, 1)

    g_running_loss = 0
    d_running_loss = 0

    for batch_idx, (real_images, real_classes) in enumerate(data_loader):
        print(batch_idx)
        if real_images.size()[0] != batch_size:
            print("batch finish")
            break
        if batch_idx > 150:
            break

        # optimize D ===========================
        d_optimizer.zero_grad()

        # calc Ls and Lc with true data
        out_label, out_class = d_model(real_images)
        L_s = b_criterion(out_label, y_real)
        L_c = c_criterion(out_class, real_classes)

        # calc Ls and Lc with fake data
        z = torch.rand((batch_size, 100))
        random_classes = torch.randint(0, 10, (batch_size, 1))
        c = torch.zeros(batch_size, 10).scatter(1, random_classes, 1)
        zc = torch.cat((z, c), 1)
        fake_images = g_model(zc)
        out_label, out_class = d_model(fake_images)
        L_s += b_criterion(out_label, y_fake)
        L_c += c_criterion(out_class, random_classes.view(-1))

        # step
        loss_d = (L_c+L_s)
        loss_d.backward()
        d_optimizer.step()
        d_running_loss += loss_d

        # optimize G =============================
        g_optimizer.zero_grad()

        # calc Ls and Lc with true data
        out_label, out_class = d_model(real_images)
        L_s = b_criterion(out_label, y_real)
        L_c = c_criterion(out_class, real_classes)

        # calc Ls and Lc with fake data
        z = torch.rand((batch_size, 100))
        random_classes = torch.randint(0, 10, (batch_size, 1))
        c = torch.zeros(batch_size, 10).scatter(1, random_classes, 1)
        zc = torch.cat((z, c), 1)
        fake_images = g_model(zc)
        out_label, out_class = d_model(fake_images)
        L_s += b_criterion(out_label, y_fake)
        L_c += c_criterion(out_class, random_classes.view(-1))

        # step
        loss_g = (L_c-L_s)
        loss_g.backward()
        g_optimizer.step()
        g_running_loss += loss_g

        print(
            f"d_loss:{loss_d/batch_size:.4}  g_loss:{loss_g/batch_size:.4}")

    d_running_loss /= len(data_loader)
    g_running_loss /= len(data_loader)

    return d_running_loss, g_running_loss


def load_data(batch_size=100, num_workers=4):
    transform = Compose((ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))))
    os.makedirs('./data', exist_ok=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    return trainloader


def generate(epoch, g_model, experiment_num):
    os.makedirs(f'./generated/ex_{experiment_num:04}', exist_ok=True)

    gen_batch_size = 10 * 2
    z = torch.rand((gen_batch_size, 100))
    c = torch.from_numpy(
        np.repeat(np.arange(10), gen_batch_size//10)).view(-1, 1)
    c = torch.zeros(gen_batch_size, 10).scatter(1, c, 1)
    zc = torch.cat((z, c), 1)
    g_model.eval()
    samples = g_model(zc)

    fname = f'./generated/ex_{experiment_num:04}/epoch_{epoch:04}.png'
    save_image(make_grid(samples, nrow=4, normalize=True), fname, nrow=4)


g_model = Generator()
d_model = Discriminator()

g_optimizer = Adam(g_model.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_optimizer = Adam(d_model.parameters(), lr=0.0001, betas=(0.5, 0.999))

b_criterion = nn.BCELoss()
c_criterion = nn.CrossEntropyLoss()


batch_size = 32
num_epochs = 100


data_loader = load_data(batch_size=batch_size)


# training ===========
history = {}
history['loss_d'] = []
history['loss_g'] = []
for epoch in range(num_epochs):
    generate(epoch=epoch, g_model=g_model, experiment_num=000)

    loss_d, loss_g = train(g_model, d_model, b_criterion, c_criterion, d_optimizer,
                           g_optimizer, data_loader, batch_size=batch_size)

    print(f'epoch {epoch+1}, loss_d: {loss_d:.4} loss_g: {loss_g:.4}')
    history['loss_d'].append(loss_d)
    history['loss_g'].append(loss_g)
