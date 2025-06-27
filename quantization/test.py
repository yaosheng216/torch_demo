from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot


MODEL_PATH = "models/flux1-kontext-dev"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


recipe = QuantizationModifier(
    targets="Linear",  # 目标层类型
    scheme="FP8_DYNAMIC",  # FP8 动态量化方案
    ignore=["lm_head"],  # 排除无需量化的层
)

oneshot(model=model, recipe=recipe)

SAVE_DIR = "models/flux1-kontext-fp8"  #
model.save_pretrained(SAVE_DIR, save_compressed=True)  
tokenizer.save_pretrained(SAVE_DIR)
