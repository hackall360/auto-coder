# 🏃 How to run LFM2

## 1. Transformers

To run LFM2, you need to install Hugging Face [`transformers`](https://github.com/huggingface/transformers) v4.55 or a more recent version as follows:

```bash
pip install -U transformers
```

Here is an example of how to generate an answer with transformers in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM2-2.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

# <|startoftext|><|im_start|>user
# What is C. elegans?<|im_end|>
# <|im_start|>assistant
# C. elegans, also known as Caenorhabditis elegans, is a small, free-living
# nematode worm (roundworm) that belongs to the phylum Nematoda.
```

You can directly run and test the model with this [Colab notebook](https://colab.research.google.com/drive/1_q3jQ6LtyiuPzFZv7Vw8xSfPU5FwkKZY?usp=sharing).

## 2. vLLM

You need to install [`vLLM`](https://github.com/vllm-project/vllm) v0.10.2 or a more recent version as follows:

```bash
uv pip install vllm==0.10.2 --extra-index-url https://wheels.vllm.ai/0.10.2/ --torch-backend=auto
```

Here is an example of how to use it for inference:

```python
from vllm import LLM, SamplingParams

prompts = [
    "What is C. elegans?",
    "Say hi in JSON format",
    "Define AI in Spanish"
]
sampling_params = SamplingParams(temperature=0.3, min_p=0.15, repetition_penalty=1.05)

llm = LLM(model="LiquidAI/LFM2-2.6B")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 3. llama.cpp

You can run LFM2 with llama.cpp using its [GGUF checkpoint](https://huggingface.co/LiquidAI/LFM2-2.6B-GGUF). Find more information in the model card.

