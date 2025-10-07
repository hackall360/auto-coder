# 📄 Model details

Due to their small size, **we recommend fine-tuning LFM2 models on narrow use cases** to maximize performance. They are particularly suited for agentic tasks, data extraction, RAG, creative writing, and multi-turn conversations. However, we do not recommend using them for tasks that are knowledge-intensive or require programming skills.

| Property            | [**LFM2-350M**](https://huggingface.co/LiquidAI/LFM2-350M) | [**LFM2-700M**](https://huggingface.co/LiquidAI/LFM2-700M) | [**LFM2-1.2B**](https://huggingface.co/LiquidAI/LFM2-1.2B) | [**LFM2-2.6B**](https://huggingface.co/LiquidAI/LFM2-2.6B) |
| ------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| **Parameters**      | 354,483,968                   | 742,489,344                   | 1,170,340,608                 | 2,569,272,320                 |
| **Layers**          | 16 (10 conv + 6 attn)         | 16 (10 conv + 6 attn)         | 16 (10 conv + 6 attn)         | 30 (22 conv + 8 attn)         |
| **Context length**  | 32,768 tokens                 | 32,768 tokens                 | 32,768 tokens                 | 32,768 tokens                 |
| **Vocabulary size** | 65,536                        | 65,536                        | 65,536                        | 65,536                        |
| **Precision**       | bfloat16                      | bfloat16                      | bfloat16                      | bfloat16                      |
| **Training budget** | 10 trillion tokens            | 10 trillion tokens            | 10 trillion tokens            | 10 trillion tokens            |
| **License**         | LFM Open License v1.0         | LFM Open License v1.0         | LFM Open License v1.0         | LFM Open License v1.0         |

**Supported languages**: English, Arabic, Chinese, French, German, Japanese, Korean, and Spanish.

**Generation parameters**: We recommend the following parameters:
* `temperature=0.3`
* `min_p=0.15`
* `repetition_penalty=1.05`

**Reasoning**: LFM2-2.6B is the only model in this family to use dynamic hybrid reasoning (traces between `<think>` and `</think>` tokens) for complex or multilingual prompts.

**Chat template**: LFM2 uses a ChatML-like chat template as follows:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
It's a tiny nematode that lives in temperate soil environments.<|im_end|>
```

You can automatically apply it using the dedicated [`.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_templating#applychattemplate) function from Hugging Face transformers.

**Tool use**: It consists of four main steps:
1. **Function definition**: LFM2 takes JSON function definitions as input (JSON objects between `<|tool_list_start|>` and `<|tool_list_end|>` special tokens), usually in the system prompt
2. **Function call**: LFM2 writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer.
3. **Function execution**: The function call is executed and the result is returned (string between `<|tool_response_start|>` and `<|tool_response_end|>` special tokens), as a "tool" role.
4. **Final answer**: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.

Here is a simple example of a conversation using tool use:

```
<|startoftext|><|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
<|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

**Architecture**: Hybrid model with multiplicative gates and short convolutions: 10 double-gated short-range LIV convolution blocks and 6 grouped query attention (GQA) blocks.

**Pre-training mixture**: Approximately 75% English, 20% multilingual, and 5% code data sourced from the web and licensed materials.

**Training approach**:
* Very large-scale SFT on 50% downstream tasks, 50% general domains
* Custom DPO with length normalization and semi-online datasets
* Iterative model merging

