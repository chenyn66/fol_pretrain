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

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--symbolic", action="store_true")
    args = parser.parse_args()

    all_result = []

    

    pretrain_data = []
    for t in ['adj']:
        for i in range(args.depth):
            pretrain_data.append(data.SYLLO(t, num_samples=(i+1)*1000, depth=i+1, symbolic=args.symbolic))

    pretrain_tester = []
    for t in ['adj']:
        for i in range(args.depth):
            pretrain_tester.append(data.SYLLO(t, num_samples=(i+1)*100, depth=i+1, symbolic=args.symbolic))

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    pre_model = syllo_finetune.LMCLS('roberta-large')

    pretrain_data = torch.utils.data.ConcatDataset(pretrain_data)
    pretrain_tester = torch.utils.data.ConcatDataset(pretrain_tester)

    train_loader = torch.utils.data.DataLoader(pretrain_data, batch_size=128, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, True))
    test_loader = torch.utils.data.DataLoader(pretrain_tester, batch_size=128, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, True))
    pre_model, acc = syllo_finetune.train(pre_model, train_loader, test_loader=test_loader, epoch=2, pbar=True, verbose=False)
    print(f'Pretrain accuracy: {acc}')



    for i in range(7,21):
        test_data = data.SYLLO('adj', num_samples=5000, depth=i)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=syllo_finetune.collate_fn(tokenizer, True))
        syllo_finetune.eval_acc(pre_model, test_loader)


        all_result.append((i, acc))

    print(all_result)
    json.dump(all_result, open(f'results/comp_{args.depth}.json', 'w'))