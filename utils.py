import json
import os
import struct

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import trange


def raw_to_single_text(path, out):
    output = open(out, 'w+')
    for root, _, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                output.write(data['article'].encode('ascii', 'ignore').decode('utf-8').replace('[]', '') + '\n')
                for q in data['questions'][:-2]:
                    output.write(q[0].encode('ascii', 'ignore').decode('utf-8') + '\n')
                    output.write('\n'.join(q[1]).encode('ascii', 'ignore').decode('utf-8') + '\n')
                output.write('\n')

    output.close()


def raw_to_text(path, out):
    idx = 0
    for i in trange(180):
        with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, q in enumerate(data['questions'][:-2]):
            with open(os.path.join(out, f'sgt_{idx}.txt'), 'w+') as output:
                output.write(data['article'].encode('ascii', 'ignore').decode('utf-8').replace('[]', '') + '\n')
                output.write(q[0].encode('ascii', 'ignore').decode('utf-8') + '\n')
                output.write('\n'.join(q[1]).encode('ascii', 'ignore').decode('utf-8') + '\n')

            idx += 1


def pack_answers(path, out):
    """
    打包单选题答案
    """
    output = open(out, 'wb+')
    for i in range(180):
        with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
            print(os.path.join(path, f'{i}.json'))
            data = json.load(f)
            for q in data['questions'][:-2]:
                # output.write(struct.pack('>I', q[2]))
                output.write(q[2].to_bytes(1))
    output.close()


def train():
    files = ['data/processed/single_text.txt']

    _tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    trainer = BpeTrainer(special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    _tokenizer.pre_tokenizer = Whitespace()

    _tokenizer.train(files=files, trainer=trainer)
    _tokenizer.save('./tokenizer/BPE.json')


def get_max_len(path, tokenizer):
    max_ = 0
    for root, _, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                s = f.read()
            max_ = max(max_, len(tokenizer.encode(s).ids))
    return max_


def tokenize(path, out, _tokenizer: Tokenizer, n=1971, seq_l=1500):
    PAD_ID = _tokenizer.token_to_id('[PAD]')
    CLS_ID = _tokenizer.token_to_id('[CLS]')
    SEP_ID = _tokenizer.token_to_id('[SEP]')
    # print(_tokenizer.get_vocab_size())
    for k in trange(n):
        with open(os.path.join(path, f'sgt_{k}.txt'), 'r') as f:
            s = f.read()
        tks = _tokenizer.encode(s).ids
        padding = seq_l - len(tks) - 2
        tks.insert(0, CLS_ID)
        tks.append(SEP_ID)
        for i in range(padding):
            tks.append(PAD_ID)
        with open(os.path.join(out, f'tks_{k}.bin'), 'wb+') as f:
            f.write(struct.pack('>1500H', *tks))


def raw_to_single_bin(path, out, _tokenizer: Tokenizer, n_json, article_l=1024, question_l=256):
    output = open(out, 'wb+')
    c = 0
    PAD_ID = _tokenizer.token_to_id('[PAD]')
    CLS_ID = _tokenizer.token_to_id('[CLS]')
    SEP_ID = _tokenizer.token_to_id('[SEP]')
    for i in trange(n_json):
        with open(os.path.join(path, f'{i}.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            article = _tokenizer.encode(data['article']).ids
            padding = article_l - len(article) - 2
            article.insert(0, CLS_ID)
            article.append(SEP_ID)
            for p in range(padding):
                article.append(PAD_ID)
            for q in data['questions'][:-2]:
                question = _tokenizer.encode(q[0] + '\n' + '\n'.join(q[1])).ids
                padding = question_l - len(question) - 2
                question.insert(0, CLS_ID)
                question.append(SEP_ID)
                for p in range(padding):
                    question.append(PAD_ID)
                output.write(
                    struct.pack(f'>{article_l}H', *article) +
                    struct.pack(f'>{question_l}H', *question)
                )

                c += 1
    output.write(struct.pack('>I', c))
    print(f'{c} processed.')


tk = Tokenizer.from_file('tokenizer/BPE.json')
raw_to_single_bin('data/raw', 'data/processed/tokens.bin', tk, 180)
