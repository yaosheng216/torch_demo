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
        self.layers = torch.nn.ModuleDict()

        # 动态创建各层
        for key, tensor in model_weights.items():
            if key.endswith(".weight") and "linear" not in key:
                path_parts = key.split('.')
                current = self

                for part in path_parts[:-2]:
                    if part not in current._modules:
                        current.add_module(part, torch.nn.ModuleDict())
                    current = current._modules[part]

                if "proj" in key or "lin" in key:
                    layer_name = path_parts[-2]
                    in_features = tensor.size(1)
                    out_features = tensor.size(0)
                    linear = torch.nn.Linear(in_features, out_features)
                    linear.weight.data = tensor

                    bias_key = key.replace(".weight", ".bias")
                    if bias_key in model_weights:
                        linear.bias.data = model_weights[bias_key]

                    current.add_module(layer_name, linear)

                elif "norm" in key:
                    layer_name = path_parts[-2]
                    normalized_shape = tensor.size(0)
                    norm = torch.nn.LayerNorm(normalized_shape)
                    norm.weight.data = tensor
                    current.add_module(layer_name, norm)
        print('layers :', self.layers.values)
        self.linear.weight.data = self.layers  # 加载权重

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

calib_dataset = [
    ...]
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
    "flux1-kontext-fp8_e4m3.safetensors",
    quantization_config=quant_config
)
