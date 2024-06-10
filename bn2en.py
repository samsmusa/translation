import torch
from normalizer import normalize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_bn_en").to(device)
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_bn_en", use_fast=False)


def clean_text(text):
    tags_to_remove = ["<pad>", "</s>"]
    for tag in tags_to_remove:
        text = text.replace(tag, "")
    return text.strip()


def chunk_text(text):
    return text.split("।")


def model_output(texts: List[str]) -> str:
    normalized_texts = [normalize(text) for text in texts]
    inputs = tokenizer(
        normalized_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    generated_tokens = model.generate(inputs["input_ids"])
    response = tokenizer.batch_decode(generated_tokens)[0]
    out = ". ".join(response)
    return str(out)


def translate(text):
    chunks = chunk_text(text)
    response = model_output(chunks)

    return clean_text(response)
