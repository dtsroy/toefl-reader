import struct

import torch
from torch.utils.data import DataLoader, Dataset
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

        a = ARTICLE_L * 2
        q = QUESTION_L * 2

        with open(qts, 'rb') as f:
            bin_tks = f.read()
            for i in range(s, e):
                self.data.append(
                    (
                        torch.tensor(
                            list(struct.unpack(f'>{ARTICLE_L}H',
                                               bin_tks[(start := i * (a + q + 4)):start + a])),
                            dtype=torch.int64
                        ).to(device),
                        torch.tensor(
                            list(struct.unpack(f'>{QUESTION_L}H',
                                               bin_tks[start + a:start + a + q])),
                            dtype=torch.int64
                        ).to(device),
                        torch.tensor(
                            (l := list(struct.unpack('>2H',
                                                     bin_tks[start + a + q:start + a + q + 4])))[0],
                            dtype=torch.int64
                        ).unsqueeze(-1).to(device),
                        torch.tensor(
                            l[1],
                            dtype=torch.int64
                        ).unsqueeze(-1).to(device)
                    )
                )
        print('Successfully loaded data')

        self.amount = len(self.data)

    def __len__(self):
        return self.amount

    def __getitem__(self, item):
        return self.data[item], self.ans_tensor[item]


def get_dataset(qts, ans, device):
    s1 = ReaderDataset(qts, ans, device)
    s2 = ReaderDataset(qts, ans, device, mode='eval')
    return s1, s2


def get_loader(qts, ans, device):
    s1, s2 = get_dataset(qts, ans, device)
    return DataLoader(s1, batch_size=50, shuffle=True), DataLoader(s2, batch_size=20, shuffle=False)
