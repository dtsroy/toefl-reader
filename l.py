# 导入tokenizer库的类和模块。
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# 用于训练tokenizer的训练数据集文件路径。
path_en = [str(file) for file in Path('./dataset-en').glob("**/*.txt")]
path_my = [str(file) for file in Path('./dataset-my').glob("**/*.txt")]

# [ 创建源语言tokenizer - 英语 ].
# 创建额外的特殊标记，如 [UNK] - 表示未知词，[PAD] - 用于维持模型序列长度相同的填充标记。
# [CLS] - 表示句子开始的标记，[SEP] - 表示句子结束的标记。
tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
trainer_en = BpeTrainer(min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

# 基于空格分割标记。
tokenizer_en.pre_tokenizer = Whitespace()

# Tokenizer训练在步骤1中创建的数据集文件。
tokenizer_en.train(files=path_en, trainer=trainer_en)

# 为将来使用保存tokenizer。
tokenizer_en.save("./tokenizer_en/tokenizer_en.json")

# [ 创建目标语言tokenizer - 马来语 ].
tokenizer_my = Tokenizer(BPE(unk_token="[UNK]"))
trainer_my = BpeTrainer(min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer_my.pre_tokenizer = Whitespace()
tokenizer_my.train(files=path_my, trainer=trainer_my)
tokenizer_my.save("./tokenizer_my/tokenizer_my.json")

tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer_en.json")
tokenizer_my = Tokenizer.from_file("./tokenizer_my/tokenizer_my.json")

# 获取两个tokenizer的大小。
source_vocab_size = tokenizer_en.get_vocab_size()
target_vocab_size = tokenizer_my.get_vocab_size()

# 定义token-ids变量，我们需要这些变量来训练模型。
CLS_ID = torch.tensor([tokenizer_my.token_to_id("[CLS]")], dtype=torch.int64).to(device)
SEP_ID = torch.tensor([tokenizer_my.token_to_id("[SEP]")], dtype=torch.int64).to(device)
PAD_ID = torch.tensor([tokenizer_my.token_to_id("[PAD]")], dtype=torch.int64).to(device)

import torch


# 此类以原始数据集和max_seq_len（整个数据集中序列的最大长度）为参数。
class EncodeDataset(Dataset):

    def __init__(self, raw_dataset, max_seq_len):
        super().__init__()

        self.raw_dataset = raw_dataset

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        # 获取给定索引处的原始文本，其包含源文本和目标文本对。

        raw_text = self.raw_dataset[index]

        # 分离文本为源文本和目标文本，稍后将用于编码。

        source_text = raw_text["en"]

        target_text = raw_text["ms"]

        # 使用源 tokenizer（tokenizer_en）对源文本进行编码，使用目标 tokenizer（tokenizer_my）对目标文本进行编码。

        source_text_encoded = torch.tensor(tokenizer_en.encode(source_text).ids, dtype=torch.int64).to(device)

        target_text_encoded = torch.tensor(tokenizer_my.encode(target_text).ids, dtype=torch.int64).to(device)

        # 为了训练模型，每个输入序列的序列长度都应等于 max seq length。

        # 因此，如果长度少于 max_seq_len，则会向输入序列添加额外的填充数量。

        num_source_padding = self.max_seq_len - len(source_text_encoded) - 2

        num_target_padding = self.max_seq_len - len(target_text_encoded) - 1

        encoder_padding = torch.tensor([PAD_ID] * num_source_padding, dtype=torch.int64).to(device)

        decoder_padding = torch.tensor([PAD_ID] * num_target_padding, dtype=torch.int64).to(device)

        # encoder_input 的第一个令牌为句子开始 - CLS_ID，后面是源编码，然后是句子结束令牌 - SEP。

        # 为了达到所需的 max_seq_len，会在末尾添加额外的 PAD 令牌。

        encoder_input = torch.cat([CLS_ID, source_text_encoded, SEP_ID, encoder_padding]).to(device)

        # decoder_input 的第一个令牌为句子开始 - CLS_ID，后面是目标编码。

        # 为了达到所需的 max_seq_len，会在末尾添加额外的 PAD 令牌。在 decoder_input 中没有句子结束令牌 - SEP。

        decoder_input = torch.cat([CLS_ID, target_text_encoded, decoder_padding]).to(device)

        # target_label 的第一个令牌为目标编码，后面是句子结束令牌 - SEP。在目标标签中没有句子开始令牌 - CLS。

        # 为了达到所需的 max_seq_len，会在末尾添加额外的 PAD 令牌。

        target_label = torch.cat([target_text_encoded, SEP_ID, decoder_padding]).to(device)

        # 由于在输入编码中添加了额外的填充令牌，因此在训练期间，我们不希望模型通过这个令牌进行学习，因为这个令牌中没有什么可学的。

        # 因此，在计算编码器块的 self attention 输出之前，我们将使用编码器掩码来使 padding 令牌的值为零。

        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int().to(device)

        # 我们也不希望任何令牌受到未来令牌的影响。因此，在掩蔽多头 self attention 期间实施了因果掩蔽以处理此问题。

        decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int() & causal_mask(
            decoder_input.size(0)).to(device)

        return {

            'encoder_input': encoder_input,

            'decoder_input': decoder_input,

            'target_label': target_label,

            'encoder_mask': encoder_mask,

            'decoder_mask': decoder_mask,

            'source_text': source_text,

            'target_text': target_text

        }


# 因果掩蔽会确保任何在当前令牌之后的令牌都被掩蔽，这意味着该值将被替换为-无穷大，然后在 softmax 函数中转换为零或接近零。
# 因此，模型将忽略这些值或无法从这些值中学习任何东西。
def causal_mask(size):
    # 因果掩蔽的维度（批量大小，序列长度，序列长度）
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


# 计算整个训练数据集中源和目标数据集的最大序列长度。
max_seq_len_source = 0
max_seq_len_target = 0

for data in raw_train_dataset["translation"]:
    enc_ids = tokenizer_en.encode(data["en"]).ids

    dec_ids = tokenizer_my.encode(data["ms"]).ids

    max_seq_len_source = max(max_seq_len_source, len(enc_ids))

    max_seq_len_target = max(max_seq_len_target, len(dec_ids))

    print(f'max_seqlen_source: {max_seq_len_source}')
# 530
print(f'max_seqlen_target: {max_seq_len_target}')
# 526

# 为了简化训练过程，我们只取一个 max_seq_len，并添加 20 来涵盖序列中额外令牌（如 PAD，CLS，SEP）的长度。
max_seq_len = 550

# 实例化 EncodeRawDataset 类，并创建编码的训练和验证数据集。
train_dataset = EncodeDataset(raw_train_dataset["translation"], max_seq_len)
val_dataset = EncodeDataset(raw_validation_dataset["translation"], max_seq_len)

# 为训练和验证数据集创建DataLoader包装器。稍后在训练和验证我们的语言模型时将使用此dataloader。
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, generator=torch.Generator(device='cuda'))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))


# 输入嵌入和位置编码
class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.d_model = d_model

        # 使用pytorch嵌入层模块将标记ID映射到词汇表，然后转换为嵌入矢量。

        # vocab_size是tokenizer在训练语料数据集时创建的训练数据集的词汇表大小。

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        # 除了将输入序列提供给嵌入层外，还对嵌入层输出进行了额外的乘法运算，以归一化嵌入层输出。

        embedding_output = self.embedding(input) * math.sqrt(self.d_model)

        return embedding_output


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        # 创建与嵌入向量形状相同的矩阵。

        pe = torch.zeros(max_seq_len, d_model)

        # 计算PE函数的位置部分。

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算PE函数的除数部分。注意除数部分的表达式略有不同，因为这种指数函数似乎效果更好。

        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(10000) / d_model)

        # 用正弦和余弦数学函数的结果填充奇数和偶数矩阵值。

        pe[:, 0::2] = torch.sin(pos * div_term)

        pe[:, 1::2] = torch.cos(pos * div_term)

        # 由于我们期望以批次的形式输入序列，因此在0位置添加了额外的batch_size维度。

        pe = pe.unsqueeze(0)

    def forward(self, input_embdding):
        # 将位置编码与输入嵌入向量相加。

        input_embdding = input_embdding + (self.pe[:, :input_embdding.shape[1], :]).requires_grad_(False)

        # 执行dropout以防止过拟合。

        return self.dropout(input_embdding)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()

        # 定义dropout以防止过拟合

        self.dropout = nn.Dropout(dropout_rate)

        # 引入权重矩阵，所有参数都可学习

        self.W_q = nn.Linear(d_model, d_model)

        self.W_k = nn.Linear(d_model, d_model)

        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by number of heads"

        #  d_k是每个分割的自注意力头的新维度

        self.d_k = d_model // num_heads

    def forward(self, q, k, v, encoder_mask=None):
        # 我们将使用多个序列的批处理来Parallel训练我们的模型，因此我们需要在形状中包括batch_size。

        # query、key和value是通过将相应的权重矩阵与输入嵌入相乘来计算的。

        # 形状变化：q(batch_size, seq_len, d_model) @ W_q(d_model, d_model) => query(batch_size, seq_len, d_model) [key和value的情况相同]。

        query = self.W_q(q)

        key = self.W_k(k)

        value = self.W_v(v)

        # 将query、key和value分割成多个head。d_model在8个头中分割为d_k。

        # 形状变化：query(batch_size, seq_len, d_model) => query(batch_size, seq_len, num_heads, d_k) -> query(batch_size,num_heads, seq_len,d_k) [key和value的情况相同]。

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # :: SELF ATTENTION BLOCK STARTS ::

        # 计算attention_score以查找query与key本身及序列中所有其他嵌入的相似性或关系。

        # 形状变化：query(batch_size,num_heads, seq_len,d_k) @ key(batch_size,num_heads, seq_len,d_k) => attention_score(batch_size,num_heads, seq_len,seq_len)。

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 如果提供了mask，则需要根据mask值修改attention_score。参见第4点的详细信息。

        if encoder_mask is not None:

            attention_score = attention_score.masked_fill(encoder_mask == 0, -1e9)

            # softmax函数计算所有attention_score的概率分布。它为较高的attention_score分配较高的概率值。这意味着更相似的令牌获得较高的概率值。

            # 形状变化：与attention_score相同

        attention_weight = torch.softmax(attention_score, dim=-1)

        if self.dropout is not None:

            attention_weight = self.dropout(attention_weight)

        # 自注意力块的最后一步是，将attention_weight与值嵌入向量相乘。

        # 形状变化：attention_score(batch_size,num_heads, seq_len,seq_len) @  value(batch_size,num_heads, seq_len,d_k) => attention_output(batch_size,num_heads, seq_len,d_k)

        attention_output = attention_score @ value

        # :: SELF ATTENTION BLOCK ENDS ::

        # 现在，所有head都将组合回一个head

        # 形状变化：attention_output(batch_size,num_heads, seq_len,d_k) => attention_output(batch_size,seq_len,num_heads,d_k) => attention_output(batch_size,seq_len,d_model)

        attention_output = attention_output.transpose(1, 2).contiguous().view(attention_output.shape[0], -1,
                                                                              self.num_heads * self.d_k)

        # 最后attention_output将与输出权重矩阵相乘以获得最终的多头注意力输出。

        # multihead_output的形状与嵌入输入相同

        # multihead_output的形状与嵌入输入相同

        multihead_output = self.W_o(attention_output)

        return multihead_output
