import torch
from normalizer import normalize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List

import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

bn_en_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_bn_en").to(device)
bn_en_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_bn_en", use_fast=False)

en_bn_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_en_bn").to(device)
en_bn_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_en_bn", use_fast=False)


def clean_text(text):
    tags_to_remove = ["<pad>", "</s>"]
    for tag in tags_to_remove:
        text = text.replace(tag, "")
    return text.strip()


def clean_string(text):
    pattern = r'^[^\u0980-\u09FFa-zA-Z0-9]+|[^\u0980-\u09FFa-zA-Z0-9]+$'
    cleaned_text = re.sub(pattern, '', text)
    print(cleaned_text)
    return cleaned_text


def chunk_text(text, split_ch="।"):
    text = clean_string(text)
    return text.split(split_ch)


def bn_en_model_output(texts: List[str]) -> str:
    normalized_texts = [normalize(text) for text in texts]
    inputs = bn_en_tokenizer(
        normalized_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    generated_tokens = bn_en_model.generate(inputs["input_ids"])
    response = bn_en_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    out = ". ".join(response)
    return str(out)


def bn_en_translate(text):
    chunks = chunk_text(text)
    response = bn_en_model_output(chunks)

    return clean_text(response)


def en_bn_model_output(texts: List[str]) -> str:
    normalized_texts = [normalize(text) for text in texts]
    inputs = en_bn_tokenizer(
        normalized_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    generated_tokens = en_bn_model.generate(inputs["input_ids"])
    response = en_bn_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    out = "। ".join(response)
    return str(out)


def en_bn_translate(text):
    chunks = chunk_text(text, split_ch=".")
    response = en_bn_model_output(chunks)

    return clean_text(response)
