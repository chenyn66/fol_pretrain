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



if __name__ == '__main__':

    pretrain_data = []
    for t in ['adj', 'noun']:
        for i in range(6):
            pretrain_data.append(data.SYLLO(t, num_samples=(i+1)*1000, depth=i+1))

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    pre_model = syllo_finetune.LMCLS('roberta-large')

    pretrain_data = torch.utils.data.ConcatDataset(pretrain_data)

    train_loader = torch.utils.data.DataLoader(pretrain_data, batch_size=64, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, True))
    pre_model, _ = syllo_finetune.train(pre_model, train_loader, epoch=10, pbar=True, verbose=False)



    folio_tr = data.FOLIO()
    folio_te = data.FOLIO(split='dev')


    model = syllo_finetune.LMFOLIO('roberta-large', pre_model.roberta)
    train_loader = torch.utils.data.DataLoader(folio_tr, batch_size=64, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))
    test_loader = torch.utils.data.DataLoader(folio_te, batch_size=16, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))


    model,result = syllo_finetune.train(model, train_loader, test_loader=test_loader, epoch=75, fp16=True, 
    lr=2e-5, warmup=0.1, pbar=True, update_every=1, verbose=True, weight_decay=1.0e-8)

    print(result)