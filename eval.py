import torch
from loader import get_dataset
from model import ReaderNetwork
from loader import get_loader

with torch.no_grad():
    VOCAB_SIZE = 20820
    _model = ReaderNetwork(VOCAB_SIZE, 128, 64, 4)
    _model.load_state_dict(torch.load('model/GPU/b.pth', map_location=torch.device('cpu')))
    ds, ts = get_dataset('data/processed/tokens.bin', 'data/processed/answers.bin', torch.device('cpu'))
    a = ds[8][0]
    # print(len(ds))
    print(_model(a[0].unsqueeze(0), a[1].view(1, -1)))

