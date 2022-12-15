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

    all_result = []

    for i in range(5):

        pretrain_data = []
        for t in ['adj', 'noun']:
            for i in range(6):
                pretrain_data.append(data.SYLLO(t, num_samples=(i+1)*1000, depth=i+1))

        pretrain_tester = []
        for t in ['adj', 'noun']:
            for i in range(6):
                pretrain_tester.append(data.SYLLO(t, num_samples=(i+1)*100, depth=i+1))

        
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        pre_model = syllo_finetune.LMCLS('roberta-large')

        pretrain_data = torch.utils.data.ConcatDataset(pretrain_data)
        pretrain_tester = torch.utils.data.ConcatDataset(pretrain_tester)

        train_loader = torch.utils.data.DataLoader(pretrain_data, batch_size=128, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, True))
        test_loader = torch.utils.data.DataLoader(pretrain_tester, batch_size=128, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, True))
        pre_model, acc = syllo_finetune.train(pre_model, train_loader, test_loader=test_loader, epoch=2, pbar=True, verbose=False)
        print(f'Pretrain accuracy: {acc}')



        folio_tr = data.FOLIO()
        folio_te = data.FOLIO(split='dev')


        model = syllo_finetune.LMFOLIO('roberta-large', pre_model.roberta)
        del pre_model
        torch.cuda.empty_cache()
        train_loader = torch.utils.data.DataLoader(folio_tr, batch_size=64, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))
        test_loader = torch.utils.data.DataLoader(folio_te, batch_size=16, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, False))


        model,acc = syllo_finetune.train(model, train_loader, test_loader=test_loader, epoch=75, fp16=True, 
        lr=2e-5, warmup=0.1, pbar=True, update_every=1, verbose=True, weight_decay=1.0e-8)

        print(f'Finetune accuracy: {acc}')

        all_result.append(acc)

    print(all_result)