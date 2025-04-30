# 🧠 Multilingual Sentiment Analysis using LLaMA 3 & Unsloth

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-blue)](https://www.kaggle.com/)

This project fine-tunes the **LLaMA 3 8B Instruct** model using **Unsloth** for **multilingual sentiment classification**. It's optimized for fast training using LoRA, and is designed to run in GPU environments like **Kaggle**.



## 🚀 Features

- ✅ Efficient fine-tuning with **LoRA**
- 🌍 Supports **multilingual** data
- 🐑 Uses **LLaMA 3 8B Instruct** model
- 📦 Built with **Unsloth**, HuggingFace `transformers`, and `datasets`
- 📊 Generates submission-ready predictions


## 📁 Dataset Structure

- `train.csv`: Labeled data with sentences and sentiment labels.
- `test.csv`: Unlabeled sentences for inference.
- `sample_submission.csv`: Format for submission.



## 📦 Installation

Make sure to install `unsloth` and other dependencies. Run in a Kaggle or GPU-enabled Python environment:

```bash
pip install --upgrade unsloth
```



## 🛠️ Model Configuration

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/kaggle/input/llama-3.1/transformers/8b-instruct/2",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None
)
```



## 🧹 Data Preprocessing

- Splits data into train/test sets.
- Converts examples into chat-style prompts.
- Prepares dataset using `standardize_sharegpt`.

```python
def preprocess(sample):
    return {
        "conversations": [
            {"from": "user", "value": sample["sentence"]},
            {"from": "assistant", "value": sample["label"]}
        ]
    }
```



## 🏋️ Training

Model fine-tuning is handled via Hugging Face’s `SFTTrainer` and LoRA config:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed_ds,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=25,
        learning_rate=2e-4,
        output_dir="outputs",
        logging_steps=5,
        fp16=True,
    ),
)
```



## 📈 Inference

Generates predictions for test sentences using the fine-tuned model:

```python
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=32)
```

Extracts the assistant's response using regex:

```python
import re
match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)
```



## 📤 Submission

```python
submission.to_csv("submission.csv", index=False)
```



## 📚 References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Meta LLaMA 3](https://ai.meta.com/llama/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)



## ⚠️ Notes

- Be sure to mount the required model and dataset files if running locally.
- The training steps are minimal for demonstration—adjust `max_steps` or `num_train_epochs` for full training.



## 📄 License

This project is licensed under the MIT License 
