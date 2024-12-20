import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 示例输入
N = 2  # 批量大小
L = 5  # 序列长度
d_model = 16  # 嵌入维度
d_ff = 32  # 前馈神经网络中间层维度

# 创建FFN模块
ffn = FeedForward(d_model, d_ff)

# 假设输入是自注意力机制的输出
input_tensor = torch.rand(N, L, d_model)

# 前向传播
output_tensor = ffn(input_tensor)
print(output_tensor.shape)  # 输出形状应为 (N, L, d_model)
