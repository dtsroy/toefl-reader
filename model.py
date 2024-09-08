import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from loader import get_loader
import torch.nn.functional as F

VOCAB_SIZE = 20820


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        # self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.T = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        # attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.T
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权的值
        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights


# class ReaderNetwork(nn.Module):
#     def __init__(self):
#         super(ReaderNetwork, self).__init__()
#         # self.embedding = nn.Embedding(VOCAB_SIZE, 128)
#         self.lstm = nn.LSTM(1, 64)
#         self.fc1 = nn.Linear(64, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 4)
#         self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         # x = self.embedding(x)
#         x, _ = self.lstm(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x

class ReaderNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_choices):
        super(ReaderNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.article_encoder = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.question_encoder = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.self_attention = SelfAttention(2 * hidden_dim)

        self.fc_merge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_choices)

        self.softmax = nn.Softmax()

    def forward(self, article_ids, question_ids):
        article_embeddings = self.embedding(article_ids)
        question_embeddings = self.embedding(question_ids)

        article_output, _ = self.article_encoder(article_embeddings)
        question_output, _ = self.question_encoder(question_embeddings)

        article_pooled, _ = self.self_attention(article_output)
        question_pooled, _ = self.self_attention(question_output)

        combined = torch.cat((article_pooled, question_pooled), dim=1)
        combined = torch.tanh(self.fc_merge(combined))

        logits = self.classifier(combined)
        score = self.softmax(logits, dim=-1)
        return score


def train(q, a):

    model = ReaderNetwork(VOCAB_SIZE, 128, 64, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    EPOCHS = 100
    bl = 1e9
    DEVICE = torch.device('cpu')

    # train_loader, test_loader = get_loader('data/processed/tokens.bin', 'data/processed/answers.bin', DEVICE)
    train_loader, test_loader = get_loader(q, a, DEVICE)

    for i in (t := trange(EPOCHS)):
        for d, y in train_loader:
            y_p = model(*d)
            loss = criterion(y_p, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lv = loss.item()

            if lv < bl:
                torch.save(model.state_dict(), 'model/b.pth')
                bl = lv
            t.set_description(f'Epoch={i}, loss={lv}, best_loss={bl}')
