import numpy as np
import json
import sys
sys.path.append('./src')
import data
import torch
import training
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


if __name__ == '__main__':
    train_dataset = data.FOL2NL(split='train')
    dev_dataset = data.FOL2NL(split='dev')
    print(len(train_dataset), len(dev_dataset))
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer.add_tokens(train_dataset.speicial_tokens)
    model.resize_token_embeddings(len(tokenizer))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=training.collate_fn(tokenizer))
    test_loder = torch.utils.data.DataLoader(dev_dataset, batch_size=8, shuffle=True, collate_fn=training.collate_fn(tokenizer))
    training.train(model, tokenizer, train_loader, test_loder, epoch=20, update_every=16)