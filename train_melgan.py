import random

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import librosa.display

from dataset import MelInversionDataset
from melgan import Generator, MultiScaleDiscriminator

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt


def visualize(real, fake, filename):
    real = real.squeeze(1).detach().to('cpu').numpy()
    fake = fake.squeeze(1).detach().to('cpu').numpy()
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(real, sr=22050)
    plt.title('Real')
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(fake, sr=22050)
    plt.title('Fake')
    fig.savefig(filename)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def train(loader, generator, discriminator, g_optimizer, d_optimizer):
    pbar = tqdm(loader)

    for i, (mel, real_wav) in enumerate(pbar):
        mel = mel.to(device) / 100
        real_wav = real_wav.to(device)

        lr = 1e-4
        # lr = 1e-4 / 100 + i / 300 * (1e-4 - 1e-4 / 100)
        # lr = min(lr, 1e-4)
        # g_optimizer.param_groups[0]['lr'] = lr
        # d_optimizer.param_groups[0]['lr'] = lr

        discriminator.zero_grad()
        requires_grad(discriminator, True)
        requires_grad(generator, False)

        fake_wav = generator(mel)
        real_predict, _ = discriminator(real_wav)
        fake_predict, _ = discriminator(fake_wav)

        disc_loss = 0
        for real, fake in zip(real_predict, fake_predict):
            loss = ((real - 1) ** 2).mean() + (fake ** 2).mean()
            disc_loss += loss

        disc_loss.backward()
        # nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        d_optimizer.step()

        generator.zero_grad()
        requires_grad(discriminator, False)
        requires_grad(generator, True)

        fake_wav = generator(mel)
        real_predict, real_feats = discriminator(real_wav)
        fake_predict, fake_feats = discriminator(fake_wav)

        gen_loss = 0
        feat_loss = 0

        for fake in real_predict:
            loss = ((fake - 1) ** 2).mean()
            gen_loss += loss

        for real, fake in zip(real_feats, fake_feats):
            loss = F.l1_loss(fake, real)
            feat_loss += loss

        loss = gen_loss + 10 * feat_loss
        loss.backward()
        # nn.utils.clip_grad_norm_(generator.parameters(), 1)
        g_optimizer.step()

        pbar.set_description(
            f'G: {gen_loss.item():.5f}; D: {disc_loss.item():.5f}; feat: {feat_loss.item():.5f}; lr: {lr:.5f}'
        )

        if i % 100 == 0:
            # print(real_wav[0])
            # print(fake_wav[0])
            # visualize(real_wav[0], fake_wav[0], f'sample/{str(i).zfill(4)}.png')
            torch.save(
                {
                    'real': real_wav.detach().to('cpu'),
                    'fake': fake_wav.detach().to('cpu'),
                },
                f'sample/{str(i).zfill(4)}.pt',
            )


if __name__ == '__main__':
    device = 'cuda'
    torch.backends.cudnn.deterministic = True

    generator = Generator(80)
    discriminator = MultiScaleDiscriminator()
    dataset = MelInversionDataset('kss.lmdb', target_len=16384)

    indices = list(range(len(dataset)))
    random.seed(17)
    random.shuffle(indices)
    train_ind = indices[200:]
    valid_ind = indices[:200]

    train_set = Subset(dataset, train_ind)
    valid_set = Subset(dataset, valid_ind)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=4)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    g_optim = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for i in range(500):
        train(train_loader, generator, discriminator, g_optim, d_optim)
        torch.save(
            {
                'g': generator.state_dict(),
                'd': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
            },
            f'checkpoint/melgan-{str(i).zfill(3)}.pt',
        )

