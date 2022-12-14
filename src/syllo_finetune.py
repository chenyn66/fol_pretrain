import torch
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import random
import data




class LMCLS(torch.nn.Module):
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(output.last_hidden_state[:,0,:]).squeeze(-1)
        loss = self.loss(logits, labels)
        return dict(loss=loss, logits=logits)


def question2text(question, tknz, shuffle_story):
    context = [i for i in question['story']]
    if shuffle_story:
        random.shuffle(context)
    return ' '.join(context) + f' {tknz.sep_token} ' + question['conclusion']


def collate_fn(tknz, shuffle_story=False):
    def func(batch):
        x = [question2text(q, tknz, shuffle_story) for q in batch]
        y = [i['label'] for i in batch]
        
        encoding = tknz(x,padding="longest",max_length=512,truncation=True,return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor(y))
    return func


def eval_acc(model, test_loader):
    model.eval()
    accelerator = Accelerator(mixed_precision='fp16')
    model, test_loader = accelerator.prepare(model, test_loader)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, input_info in enumerate(test_loader):
            pred = model(input_ids=input_info['input_ids'], attention_mask=input_info['attention_mask'], labels=input_info['labels'])['logits']
            correct += ((pred>0).float() == input_info['labels']).sum().item()
            total += input_info['input_ids'].shape[0]
    return correct/total



def train(model, train_loader, test_loader=None, epoch=1, fp16=True, lr=1e-5, warmup=0.1, pbar=True, update_every=1, verbose=True):
    torch.cuda.empty_cache()
    accelerator = Accelerator(gradient_accumulation_steps=update_every, mixed_precision='fp16' if fp16 else 'no')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup*epoch*len(train_loader)//update_every),
    num_training_steps=epoch*len(train_loader)//update_every)


    for ep in range(epoch):
        model.train()
        for batch_idx, input_info in enumerate(tqdm(train_loader) if pbar else train_loader):
            with accelerator.accumulate(model):
                loss = model(input_ids=input_info['input_ids'], attention_mask=input_info['attention_mask'], labels=input_info['labels'])['loss']
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if warmup is not None and not accelerator.optimizer_step_was_skipped and batch_idx % update_every == 0:
                    schedule.step()
                if batch_idx % 10 == 0 and verbose:
                    print(f'Epoch: {ep+1}/{epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()*update_every:.4f}')
        
        if test_loader is not None:
            print(f'Epoch: {ep+1}/{epoch}, Test Acc: {eval_acc(model, test_loader):.4f}')
    return model




def test_composition(template='noun', dmax=5, num_samples=1000, model_name="roberta-large", shuffle_story=True, n_runs=1):
    results = {d+1:[] for d in range(dmax)}
    for d in tqdm(range(dmax)):
        print(f'Train Depth: {d+1}/{dmax}')
        for i in tqdm(range(n_runs)):
            print(f'Run: {i+1}/{n_runs}')
            torch.cuda.empty_cache()
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            train_dataset = data.SYLLO(template, num_samples=num_samples, depth=d+1)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn(tokenizer, shuffle_story))
            model = LMCLS(model_name)
            model = train(model, train_loader, epoch=10, update_every=2, pbar=False, verbose=False)

            test_results = dict()
            for test_d in range(dmax):
                test_dataset = data.SYLLO(template, num_samples=num_samples, depth=test_d+1)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn(tokenizer, shuffle_story))
                test_results[test_d] = eval_acc(model, test_loader)
                print(f'Test Depth: {test_d+1}, Test Acc: {test_results[test_d]:.4f}')
            results[d+1].append(test_results)
            del model
            
    return results