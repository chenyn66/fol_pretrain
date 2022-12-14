import torch
import json
import random
import syllo_gen



'''
For finding special tokens in the dataset
sepcial_chars = set()
for x,y in train_dataset:
    for chars in x:
        if tokenizer.unk_token_id in tokenizer(chars,add_special_tokens=False).input_ids:
            sepcial_chars.add(chars)
sepcial_chars = list(sepcial_chars)
'''

'''
For GPT3 prompt

data = open('dataset/folio/folio-train.jsonl',encoding='utf-8')
all_pairs = []
for line in data:
    line = json.loads(line)
    for nl, fol in zip(line['premises'], line['premises-FOL']):
        all_pairs.append((nl, fol))


data = open('dataset/folio/folio-validation.jsonl',encoding='utf-8')
for line in data:
    line = json.loads(line)
    for nl, fol in zip(line['premises']+[line['conclusion']], line['premises-FOL']+[line['conclusion-FOL']]):
        all_pairs.append((nl, fol))

for nl, fol in all_pairs[:10]:
    print(f'Input: {fol}')
    print(f'Output: {nl}')
    print()
print(f'Input: {all_pairs[10][1]}')
'''

'''
For GPT3 finetuning
unique_sample = set()
with open('dataset/folio/gpt-finetune.jsonl','w',encoding='utf-8') as f:
    for x,y in train_dataset:
        if (x,y) in unique_sample:
            continue
        f.write(json.dumps({'prompt':x,'completion':y})+'\n')
        unique_sample.add((x,y))

    # for x,y in dev_dataset:
    #     if (x,y) in unique_sample:
    #         continue
    #     f.write(json.dumps({'prompt':x,'completion':y})+'\n')
    #     unique_sample.add((x,y))
'''


class FOL2NL(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        self.data = []
        self.speicial_tokens = ['→', '↔', '¬', '⊕', '∨', '∧', '∀', '∃']
        if split == 'dev':
            split = 'valid'
        assert split in ['train', 'valid']
        if split == 'valid':
            split = 'validation'

        with open(f'dataset/folio/folio-{split}.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                for nl, fol in zip(line['premises'], line['premises-FOL']):
                    if nl == '' or fol == '': 
                        continue
                    self.data.append((fol.replace('^','∧'), nl))
                
                if 'conclusion-FOL' in line:
                    self.data.append((line['conclusion-FOL'].replace('^','∧'), line['conclusion']))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SYLLO(torch.utils.data.Dataset):
    def __init__(self, template='adj', depth=2, num_samples=1000, symbolic=False):
        self.data = []

        assert template in ['noun', 'adj']
        if template == 'noun':
            assign_func = syllo_gen.random_assign_nouns
        else:
            assign_func = syllo_gen.random_assign_adjs
        
        if symbolic:
            assign_func = lambda x: x

        
        while len(self.data) < num_samples:
            real = random.choice([True, False])
            q, v = syllo_gen.get_syllo(depth)
            if not real:
                q = syllo_gen.negate_quesion(q)
            v = assign_func(v)
            if symbolic:
                q = syllo_gen.question2fol(q, v)
            else:
                q = syllo_gen.question2template(q, v, rand=True, noun=template=='noun')
            q['label'] = 1. if real else 0.
            self.data.append(q)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class FOLIO(torch.utils.data.Dataset):
    def __init__(self, split='train', tf_only=False, combine=False):
        self.data = []
        if split == 'dev':
            split = 'valid'
        assert split in ['train', 'valid']
        if split == 'valid':
            split = 'validation'

        
        if combine:
            with open('dataset/folio/folio-train.jsonl', 'r', encoding='utf-8') as f, open('dataset/folio/folio-validation.jsonl', 'r', encoding='utf-8') as f2:
                raw_data = f.readlines()
                raw_data += f2.readlines()
        else:
            with open(f'dataset/folio/folio-{split}.jsonl', 'r', encoding='utf-8') as f:
                raw_data = f.readlines()

        for line in raw_data:
            line = json.loads(line)
            q = {'story': line['premises'],
            'conclusion': line['conclusion']
            }
            if line['label'] == 'False':
                q['label'] = 0
            elif line['label'] == 'True':
                q['label'] = 1
            elif line['label'] == 'Unknown' or line['label'] == 'Uncertain':
                if tf_only:
                    continue
                q['label'] = 2
            else:
                raise ValueError(f'Unknown label: {line["label"]}')
            
            if tf_only:
                q['label'] = float(q['label'])

            self.data.append(q)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]




    
            