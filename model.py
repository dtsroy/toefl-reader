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

        # self.batch_norm1 = nn.BatchNorm1d(2 * hidden_dim)
        self.dropout1 = nn.Dropout(p=0.4)

        self.self_attention = SelfAttention(2 * hidden_dim)

        # self.batch_norm2 = nn.BatchNorm1d(2 * hidden_dim)
        # self.dropout2 = nn.Dropout(p=0.4)

        self.fc_merge = nn.Linear(2 * hidden_dim, hidden_dim)

        # self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(p=0.4)

        self.classifier = nn.Linear(hidden_dim, num_choices)

    def forward(self, article_ids, question_ids, article_length, question_length):
        article_embeddings = self.embedding(article_ids)
        question_embeddings = self.embedding(question_ids)

        # Create masks
        article_mask = torch.gt(article_ids, 0)
        question_mask = torch.gt(question_ids, 0)

        # Pack sequences
        packed_article = nn.utils.rnn.pack_padded_sequence(
            article_embeddings,
            article_length.flatten().cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_question = nn.utils.rnn.pack_padded_sequence(
            question_embeddings,
            question_length.flatten().cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        article_output, _ = self.article_encoder(packed_article)
        question_output, _ = self.question_encoder(packed_question)

        # Unpack sequences
        article_output, _ = nn.utils.rnn.pad_packed_sequence(article_output, batch_first=True)
        question_output, _ = nn.utils.rnn.pad_packed_sequence(question_output, batch_first=True)

        article_output = nn.functional.pad(
            article_output,
            (0, 0, 0, article_embeddings.size()[1] - article_output.size()[1]),
            value=0
        )

        question_output = nn.functional.pad(
            question_output,
            (0, 0, 0, question_embeddings.size()[1] - question_output.size()[1]),
            value=0
        )

        # Apply masks
        article_output = article_output * article_mask.unsqueeze(-1).float()
        question_output = question_output * question_mask.unsqueeze(-1).float()

        # article_output = self.batch_norm1(article_output)  # Apply Batch Norm
        # question_output = self.batch_norm1(question_output)  # Apply Batch Norm
        article_output = self.dropout1(article_output)
        question_output = self.dropout1(question_output)

        article_pooled, _ = self.self_attention(article_output)
        question_pooled, _ = self.self_attention(question_output)

        combined = torch.cat((article_pooled, question_pooled), dim=1)

        # combined = self.batch_norm2(combined)
        # combined = self.dropout2(combined)

        combined = self.fc_merge(combined)

        # combined = self.batch_norm3(combined)
        combined = self.dropout3(combined)

        combined = torch.tanh(combined)

        logits = self.classifier(combined)
        return logits[:, -1, :]


def train(q, a):
    EPOCHS = 100
    bl = 1e9
    DEVICE = torch.device('cpu')

    model = ReaderNetwork(VOCAB_SIZE, 128, 64, 4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train_loader, test_loader = get_loader(q, a, DEVICE)

    for i in (t := trange(EPOCHS)):
        for d, y in train_loader:
            # print(d[2])
            y_p = model(*d)
            loss = criterion(y_p, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lv = loss.item()
            if lv < bl:
                torch.save(model.state_dict(), 'model/b.pth')
                bl = lv
            t.set_description(f'Epoch={i+1}, loss={lv}, best_loss={bl}')
        with torch.no_grad():
            model.eval()
            k = 0
            c = 0
            for d2, y2 in test_loader:
                y2_p = model(*d2)
                k += criterion(y2_p, y2).item()
                c += 1
            print(f'Epoch={i+1}, test_avg_loss={k/c}')
        model.train()


if __name__ == '__main__':
    train('data/processed/tokens.bin', 'data/processed/answers.bin')
