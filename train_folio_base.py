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
import argparse
import json


if __name__ == '__main__':

    all_result = []

    for _ in range(5):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        folio_tr = data.FOLIO()
        folio_te = data.FOLIO(split='dev')


        model = syllo_finetune.LMFOLIO('roberta-large')
        torch.cuda.empty_cache()
        train_loader = torch.utils.data.DataLoader(folio_tr, batch_size=64, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))
        test_loader = torch.utils.data.DataLoader(folio_te, batch_size=16, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))


        model,acc = syllo_finetune.train(model, train_loader, test_loader=test_loader, epoch=75, fp16=True, 
        lr=2e-5, warmup=0.1, pbar=True, update_every=1, verbose=False, weight_decay=1.0e-8)

        print(f'Finetune accuracy: {acc}')

        all_result.append(acc)

    print(all_result)
    json.dump(all_result, open(f'./result/syllo_baseline.json', 'w'))