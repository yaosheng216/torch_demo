import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 稀疏层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 定义GRU网络
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # view:进行维度转换
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    # 初始化隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttendDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super(AttendDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_len = max_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 计算attention权重
        atten_weight = F.softmax(
            self.attn(torch.cat([embedded[0], hidden[0]], 1)),
            dim=1
        )

        att_applied = torch.bmm(
            atten_weight.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )
        output = torch.cat([embedded[0], att_applied[0]], dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, atten_weight

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


if __name__ == "__main__":
    encoder_net = EncoderRNN(5000, 256)
    decoder_net = DecoderRNN(256, 5000)
    attend_decoder_net = AttendDecoderRNN(256, 5000)

    tensor_in = torch.tensor([12, 14, 16, 18], dtype=torch.long).view(-1, 1)
    hidden_in = torch.zeros(1, 1, 256)

    encoder_out, encoder_hidden = encoder_net(tensor_in[0], hidden_in)

    print(encoder_out)
    print(encoder_hidden)

    tensor_in = torch.tensor([[100]])
    hidden_in = torch.zeros(1, 1, 256)
    encoder_out = torch.zeros(10, 256)

    out1, out2, out3 = attend_decoder_net(tensor_in, hidden_in, encoder_out)
    print(out1, out2, out3)

    out1, out2 = decoder_net(tensor_in, hidden_in)
    print(out1, out2)
