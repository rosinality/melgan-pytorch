import argparse
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import KSSDataset, TTSDataset, collate_data
from tacotron2 import Tacotron2
from scheduler import cycle_scheduler


class Config:
    pass


conf = Config()
conf.n_vocab = 256
conf.embed_dim = 512
conf.enc_kernel_size = 5
conf.enc_n_conv = 3
conf.enc_conv_dim = 512
conf.enc_lstm_dim = 512
conf.enc_n_lstm = 1
conf.dropout = 0.5

conf.n_mels = 80
conf.pre_dim = 256
conf.attention_dim = 128
conf.loc_dim = 32
conf.loc_kernel = 31
conf.dec_dim = 1024
conf.dec_n_layer = 2
conf.post_dim = 512
conf.post_kernel_size = 5
conf.n_post = 5
conf.zoneout = 0.1
conf.deterministic = False
conf.dec_max_length = 1000
conf.dec_stop_threshold = 0


def train(args, loader, model, optimizer, scheduler):
    pbar = tqdm(loader)

    model.train()

    for texts, text_len, mels, mels_len, stops in pbar:
        texts = texts.to(device)
        text_len = text_len.to(device)
        mels = mels.to(device)
        mels_len = mels_len.to(device)
        stops = stops.to(device)

        model.zero_grad()

        _, _, _, loss = model(texts, text_len, mels, mels_len, stops)
        mel_loss = loss['mel'].mean()
        mel_post_loss = loss['mel_post'].mean()
        stop_loss = loss['stop'].mean()
        loss_val = mel_loss + mel_post_loss + stop_loss
        loss_val.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        pbar.set_description(
            f'mel: {mel_loss.item():.5f}; mel post: {mel_post_loss.item():.5f}; stop: {stop_loss.item():.5f}; lr: {lr:.5f}'
        )

        if args.wandb:
            wandb.log(
                {
                    'train/mel': mel_loss.item(),
                    'train/mel post': mel_post_loss.item(),
                    'train/stop': stop_loss.item(),
                    'train/lr': lr,
                }
            )


def valid(args, loader, model):
    pbar = tqdm(loader)

    mel_loss_sum = 0
    mel_post_loss_sum = 0
    stop_loss_sum = 0
    total_sample = 0

    model.eval()

    for texts, text_len, mels, mels_len, stops in pbar:
        texts = texts.to(device)
        text_len = text_len.to(device)
        mels = mels.to(device)
        mels_len = mels_len.to(device)
        stops = stops.to(device)

        _, _, _, loss = model(texts, text_len, mels, mels_len, stops, valid=True)
        mel_loss = loss['mel'].mean().item()
        mel_post_loss = loss['mel_post'].mean().item()
        stop_loss = loss['stop'].mean().item()

        batch_size = texts.shape[0]

        mel_loss_sum += mel_loss * batch_size
        mel_post_loss_sum += mel_post_loss * batch_size
        stop_loss_sum += stop_loss * batch_size

        total_sample += batch_size

        mel_loss_mean = mel_loss_sum / total_sample
        mel_post_loss_mean = mel_post_loss_sum / total_sample
        stop_loss_mean = stop_loss_sum / total_sample

        pbar.set_description(
            f'mel: {mel_loss_mean:.5f}; mel post: {mel_post_loss_mean:.5f}; stop: {stop_loss_mean:.5f}'
        )

    if args.wandb:
        wandb.log(
            {
                'valid/mel': mel_loss_mean,
                'valid/mel post': mel_post_loss_mean,
                'valid/stop': stop_loss_mean,
            }
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--n_valid', type=int, default=200)
    parser.add_argument('--warmup', type=float, default=0.05)
    parser.add_argument('--plateau', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()

    if args.wandb:
        import wandb

        wandb.init(project='tacotron 2')

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    torch.backends.cudnn.deterministic = True

    device = 'cuda'
    model = Tacotron2(conf)
    dataset = TTSDataset('kss.lmdb')

    indices = list(range(len(dataset)))
    random.seed(17)
    random.shuffle(indices)
    train_ind = indices[200:]
    valid_ind = indices[:200]

    train_set = Subset(dataset, train_ind)
    valid_set = Subset(dataset, valid_ind)

    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_data
    )
    valid_loader = DataLoader(
        valid_set, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_data
    )

    model = model.to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr / 25, eps=1e-6, weight_decay=args.l2
    )
    scheduler = cycle_scheduler(
        optimizer,
        args.lr,
        len(train_loader) * args.epoch,
        warmup=args.warmup,
        plateau=args.plateau,
    )

    for i in range(args.epoch):
        train(args, train_loader, model, optimizer, scheduler)
        valid(args, valid_loader, model)
        torch.save(model.module.state_dict(), f'checkpoint/{str(i).zfill(3)}.pt')
