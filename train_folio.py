from transformers import RobertaTokenizer, RobertaModel
import torch
import sys
sys.path.append('./src')
import syllo_gen
import random
import json
from tqdm.autonotebook import tqdm
import data
import syllo_finetune



if __name__ == '__main__':
    folio_tr = data.FOLIO()
    folio_te = data.FOLIO(split='dev')


    model = syllo_finetune.LMFOLIO('roberta-large')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    train_loader = torch.utils.data.DataLoader(folio_tr, batch_size=64, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))
    test_loader = torch.utils.data.DataLoader(folio_te, batch_size=16, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))


    syllo_finetune.train(model, train_loader, test_loader=test_loader, epoch=75, fp16=True, 
    lr=2e-5, warmup=0.1, pbar=True, update_every=1, verbose=True, weight_decay=1.0e-8)