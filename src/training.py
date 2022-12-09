import torch
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


def collate_fn(tknz):
    def func(batch):
        x,y = [i[0] for i in batch], [i[1] for i in batch]
        
        encoding = tknz(x,padding="longest",max_length=256,truncation=True,return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        target_encoding = tknz(y,padding="longest",max_length=256,truncation=True,return_tensors="pt")
        labels = target_encoding.input_ids
        labels[labels == tknz.pad_token_id] = -100
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, text=(x,y))
    return func




def train(model, tokenizer, train_loader, test_loader=None, epoch=1, fp16=True, lr=1e-5, warmup=0.1, pbar=True, update_every=1):

    accelerator = Accelerator(gradient_accumulation_steps=update_every, mixed_precision='fp16' if fp16 else 'no')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup*epoch*len(train_loader)//update_every),
    num_training_steps=epoch*len(train_loader)//update_every)


    for ep in range(epoch):
        model.train()
        for batch_idx, input_info in enumerate(tqdm(train_loader) if pbar else train_loader):
            with accelerator.accumulate(model):
                loss = model(input_ids=input_info['input_ids'], attention_mask=input_info['attention_mask'], labels=input_info['labels']).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if warmup is not None and not accelerator.optimizer_step_was_skipped and batch_idx % update_every == 0:
                    schedule.step()
                if batch_idx % 100 == 0:
                    print(f'Epoch: {ep+1}/{epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()*update_every:.4f}')
        
        if test_loader is not None:
            model.eval()
            total_loss = 0
            total = 0
            with torch.no_grad():
                for batch_idx, input_info in enumerate(tqdm(test_loader) if pbar else test_loader):
                    loss = model(input_ids=input_info['input_ids'], attention_mask=input_info['attention_mask'], labels=input_info['labels']).loss
                    total_loss += loss.item()*input_info['input_ids'].shape[0]
                    total += input_info['input_ids'].shape[0]
                print(f'Epoch: {ep+1}/{epoch}, Test Loss: {total_loss/total:.4f}')
                for input_info in test_loader:
                    outputs = model.generate(input_info['input_ids'], attention_mask=input_info['attention_mask'], max_length=256, num_beams=4, early_stopping=True)
                    for input_id, pred, target in zip(input_info['input_ids'], outputs, input_info['text'][1]):
                        print(f'Input: {tokenizer.decode(input_id, skip_special_tokens=True)}\nPred: {tokenizer.decode(pred, skip_special_tokens=True)}\nTarget: {target}\n')
                        break
                    break
                


