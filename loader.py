import os.path

import torch
from torch.utils.data import DataLoader, Dataset
import struct
from tqdm import trange


VOCAB_SIZE = 20820
ARTICLE_L = 1024
QUESTION_L = 256


class ReaderDataset(Dataset):
    def __init__(self, qts, ans, device, mode='train', ratio=0.7):
        super(ReaderDataset, self).__init__()
        with open(ans, 'rb') as f:
            ans_data = f.read()
        self.amount = len(ans_data)

        s = 0 if mode == 'train' else int(self.amount * ratio)
        e = int(self.amount * ratio) if mode == 'train' else self.amount

        tmp = [
            torch.tensor([[1, 0, 0, 0]], dtype=torch.long),
            torch.tensor([[0, 1, 0, 0]], dtype=torch.long),
            torch.tensor([[0, 0, 1, 0]], dtype=torch.long),
            torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
        ]
        self.ans_tensor = torch.cat([tmp[i] for i in ans_data[s:e]], dim=0).to(device)

        self.data = []
        print('loading data...')

        with open(qts, 'rb') as f:
            bin_tks = f.read()
            for i in trange(s, e):
                self.data.append(
                    (
                        torch.tensor(
                            list(struct.unpack(f'>{ARTICLE_L}H',
                                               bin_tks[(start := 2 * i * (ARTICLE_L + QUESTION_L)):start + ARTICLE_L * 2])),
                            dtype=torch.int64
                        ).to(device),
                        torch.tensor(
                            list(struct.unpack(f'>{QUESTION_L}H',
                                               bin_tks[start + ARTICLE_L * 2:start + ARTICLE_L * 2 + QUESTION_L * 2])),
                            dtype=torch.int64
                        ).to(device)
                    )
                )
        print('Successfully loaded data')

        self.amount = len(self.data)

    def __len__(self):
        return self.amount

    def __getitem__(self, item):
        return self.data[item], self.ans_tensor[item]


def get_loader(qts, ans, device):
    s1 = ReaderDataset(qts, ans, device)
    s2 = ReaderDataset(qts, ans, device, mode='eval')
    return DataLoader(s1, batch_size=100, shuffle=True), DataLoader(s2, batch_size=20, shuffle=False)
