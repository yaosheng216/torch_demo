import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests

model_id = "yifeihu/TB-OCR-preview-0.1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="cuda",
  trust_remote_code=True,
  torch_dtype="auto",
  _attn_implementation='flash_attention_2',
  quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

processor = AutoProcessor.from_pretrained(model_id,
  trust_remote_code=True,
  num_crops=16
)


def phi_ocr(image_url):
    question = "Convert the text to markdown format."
    image = Image.open(requests.get(image_url, stream=True).raw)
    prompt_message = [{
        'role': 'user',
        'content': f'<|image_1|>\n{question}',
    }]

    prompt = processor.tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda")

    generation_args = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "do_sample": False
    }
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = response.split("<image_end>")[0]

    return response


test_image_url = "https://huggingface.co/yifeihu/TB-OCR-preview-0.1/resolve/main/sample_input_1.png?download=true"
response = phi_ocr(test_image_url)
print(response)
