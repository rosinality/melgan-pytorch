import pickle
from multiprocessing import Pool

import numpy as np
import lmdb
import librosa
from tqdm import tqdm

from dataset import KSSDataset


def read(data):
    sr = 22050
    preemphasis = 0.97
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    mel_min = 1e-5
    wav_trim = 1
    trim_top_db = 23

    path, text = data

    y, sr = librosa.load(path, sr=sr, mono=True)

    if preemphasis is not None:
        y = librosa.effects.preemphasis(y, preemphasis)

    y = y[wav_trim:]

    if trim_top_db > 0:
        y, _ = librosa.effects.trim(
            y, top_db=trim_top_db, frame_length=n_fft, hop_length=hop_length
        )

    mels = librosa.feature.melspectrogram(
        y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    if mel_min is not None:
        mels = np.clip(mels, mel_min, None)

    mels = np.log(mels)

    record = pickle.dumps((y, mels, text))

    return record


if __name__ == '__main__':
    root = '/root/data/kss'
    filename = 'kss.lmdb'
    sr = 22050
    preemphasis = 0.97

    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    dset = KSSDataset(root)

    def data_iter(dset):
        for i in range(len(dset)):
            yield dset[i]

    with Pool(processes=8) as pool, lmdb.open(
        filename, map_size=1024 ** 4, readahead=False
    ) as env:
        for i, record in enumerate(tqdm(pool.imap(read, data_iter(dset)))):
            with env.begin(write=True) as txn:
                txn.put(str(i).encode('utf-8'), record)

        with env.begin(write=True) as txn:
            txn.put(b'length', str(len(dset)).encode('utf-8'))
            txn.put(b'config', pickle.dumps({'sr': sr, 'preemphasis': preemphasis}))
