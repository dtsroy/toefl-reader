import torch
from loader import get_dataset
from model import ReaderNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm

with torch.no_grad():
    VOCAB_SIZE = 20820
    _model = ReaderNetwork(VOCAB_SIZE, 128, 64, 4)
    _model.load_state_dict(torch.load('model/GPU/f_with_dropout.pth', map_location=torch.device('cpu')))
    _model.eval()
    ds, ts = get_dataset('data/processed/tokens.bin', 'data/processed/answers.bin', torch.device('cpu'))
    lo = DataLoader(ds, batch_size=1, shuffle=False)
    j = 0
    for c, (d, p) in tqdm(list(enumerate(lo))[:601]):
        r = torch.nn.functional.softmax(_model(*d)[0], 0)
        r = [i.item() for i in list(r)]
        if r.index(max(r)) == [i.item() for i in list(p[0])].index(1):
            # print('ok!')
            j += 1
    print(j, j / (c+1))
