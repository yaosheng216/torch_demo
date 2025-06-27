from neural_compressor import PostTrainingQuantConfig, quantization
from safetensors import safe_open
from compressed_tensors import save_compressed, QuantizationConfig
import torch


# 加载 safetensors 模型
model_weights = {}
with safe_open("flux1-kontext-dev.safetensors", framework="pt") as f:
    for key in f.keys():
        model_weights[key] = f.get_tensor(key)


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256)
        self.linear.weight.data = model_weights["linear.weight"]  # 加载权重

    def forward(self, x):
        return self.linear(x)


model = CustomModel().eval()

# 配置 FP8 量化 (推荐 E4M3 格式)
fp8_config = PostTrainingQuantConfig(
    approach="post_training_static_quant",
    backend="fx",  # 使用 PyTorch FX 模式
    quant_format="fp8",  # 指定 FP8 格式
    recipes={"fp8_format": "E4M3"}
)

calib_dataset = [...]
q_model = quantization.fit(
    model=model,
    conf=fp8_config,
    calib_dataloader=torch.utils.data.DataLoader(calib_dataset, batch_size=32)
)

# 保存为 safetensors 格式
quant_config = QuantizationConfig.from_dict({
    "quantization": {
        "weights": {
            "format": "fp8_e4m3",  # 与量化配置一致
            "bits": 8,
            "group_size": 128  # 可选分组量化
        }
    }
})
save_compressed(
    q_model.model.state_dict(),
    "quantized_model.safetensors",
    quantization_config=quant_config
)
