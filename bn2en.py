import torch
from normalizer import normalize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def model_output(text: str) -> str:
    input_ids = tokenizer(normalize(text), return_tensors="pt").input_ids
    generated_tokens = model.generate(input_ids)
    response = tokenizer.batch_decode(generated_tokens)[0]
    return str(response)


def translate(text):
    response = ""
    chunks = chunk_text(text)
    for chunk in chunks:
        response = response + model_output(chunk)

    return clean_text(response)
