import os
import pickle
import random

import torch
import lmdb
import numpy as np
import librosa
from torch.utils.data import Dataset


class KSSDataset:
    def __init__(self, path):
        self.root = path

        with open(os.path.join(path, 'transcript.v.1.3.txt')) as f:
            data = f.readlines()

        transcript = []
        for line in data:
            line = line.strip().split('|')
            path, text = line[0], line[1]
            transcript.append((path, text))

        self.transcript = transcript
        self.n_vocab = 128 + 19 + 21 + 28  # or 256?

    def decompose(self, char):
        HANGUL_BASE = 0xAC00

        code = ord(char) - HANGUL_BASE

        initial, resid = divmod(code, 21 * 28)
        vowel = resid // 28
        final = code % 28

        return initial, vowel, final

    def is_hangul(self, char):
        if 0xAC00 <= ord(char) <= 0xD7AF:
            return True

        else:
            return False

    def text_to_id(self, text):
        ids = []

        for ch in text:
            if self.is_hangul(ch):
                init, vow, fin = self.decompose(ch)
                init_id = 128 + init
                vow_id = 128 + 30 + vow
                fin_id = 128 + 30 + 30 + fin

                ids.extend((init_id, vow_id, fin_id))

            else:
                id = ord(ch)

                if id > 127:
                    id = 1

                ids.append(id)

        return ids

    def __len__(self):
        return len(self.transcript)

    def __getitem__(self, index):
        path, text = self.transcript[index]
        ids = self.text_to_id(text)

        return os.path.join(self.root, path), ids


class TTSDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'length').decode('utf-8'))
            config = pickle.loads(txn.get(b'config'))
            self.sr = config['sr']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            record = txn.get(key)

            wav, mels, ids = pickle.loads(record)

        return torch.from_numpy(mels), torch.tensor(ids, dtype=torch.int64)


class MelInversionDataset(Dataset):
    def __init__(
        self, path, target_len, n_fft=1024, hop_length=256, n_mels=80, mel_min=1e-5
    ):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'length').decode('utf-8'))
            config = pickle.loads(txn.get(b'config'))
            self.sr = config['sr']

        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_min = mel_min

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            record = txn.get(key)

            y, _, _ = pickle.loads(record)

        if y.shape[0] >= self.target_len:
            max_id = y.shape[0] - self.target_len
            start = random.randint(0, max_id)
            y = y[start : start + self.target_len]

        else:
            y = np.pad(y, (0, self.target_len - y.shape[0]), 'constant')

        mels = librosa.feature.melspectrogram(
            y[:-1],
            self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mels = mels[:, : y.shape[0] // 256]

        if self.mel_min is not None:
            mels = np.clip(mels, self.mel_min, None)

        mels = np.log(mels)

        mels = torch.from_numpy(mels)
        wav = torch.from_numpy(y).unsqueeze(0)

        return mels, wav


def collate_data(batch):
    max_mel_len = max(b[0].shape[1] for b in batch)
    max_text_len = max(len(b[1]) for b in batch)

    batch_size = len(batch)
    n_mels = batch[0][0].shape[0]

    mels = torch.zeros(batch_size, n_mels, max_mel_len, dtype=torch.float32)
    texts = torch.zeros(batch_size, max_text_len, dtype=torch.int64)
    stops = torch.zeros(batch_size, max_mel_len, dtype=torch.float32)

    text_lengths = torch.zeros(batch_size, dtype=torch.int64)
    mels_lengths = torch.zeros(batch_size, dtype=torch.int64)

    for i, b in enumerate(batch):
        mel, text = b
        mel_len = mel.shape[1]
        text_len = len(text)

        mels[i, :, :mel_len] = mel
        texts[i, :text_len] = text
        stops[i, mel_len - 1 :] = 1

        text_lengths[i] = text_len
        mels_lengths[i] = mel_len

    return texts, text_lengths, mels, mels_lengths, stops
