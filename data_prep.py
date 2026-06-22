import os

import numpy
import tiktoken
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")
enc = tiktoken.get_encoding("gpt2")
train = ds["train"]
val = ds["validation"]


def encode(example):
    text = example["text"]
    tokens = enc.encode_ordinary(text)
    tokens.append(enc.eot_token)
    return {"ids": tokens, "len": len(tokens)}


train_map = train.map(encode, remove_columns=["text"], num_proc=12)
val_map = val.map(encode, remove_columns=["text"], num_proc=12)

train_total = numpy.sum(train_map["len"])
val_total = numpy.sum(val_map["len"])

os.makedirs("data", exist_ok=True)
train_path = os.path.join("data", "train.bin")
val_path = os.path.join("data", "val.bin")

train_arr = numpy.memmap(
    train_path, dtype=numpy.uint16, mode="w+", shape=(train_total,)
)
val_arr = numpy.memmap(val_path, dtype=numpy.uint16, mode="w+", shape=(val_total,))

train_offset = 0
val_offset = 0

for example in train_map:
    ids = example["ids"]
    length = example["len"]

    train_arr[train_offset : train_offset + length] = ids
    train_offset += length
train_arr.flush()

for example in val_map:
    ids = example["ids"]
    length = example["len"]

    val_arr[val_offset : val_offset + length] = ids
    val_offset += length
val_arr.flush()

print(val_offset == val_total)
print(train_total == train_offset)
