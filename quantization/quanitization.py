import torch


model = torch.load('/mnt/data/yaosheng/test/models/test1.pth')
# 动态量化
model_int8 = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
'''
    tensor([[-0.8211, 0.1416, 0.9627],
        [-1.9537, 0.5380, 1.6989],
        [-3.6243, 1.7555, 1.6423], size=[3, 3], dtype=torch.qint8,
        quantization_scheme=torch.per_tensor_affine, scale=0.028314674273133278,
        zero_point=0)
'''
# 打印量化后的模型权重(int8)
print(torch.int_repr(model_int8.linear.weight))
# 反量化后的模型权重
print(model_int8.linear.weight())
