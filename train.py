import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from torch.optim import AdamW
import tiktoken
import time
import inspect
import os
from gpt2_model import GPT, GPTConfig

#########################

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

num_return_sequences = 5
max_length = 30

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 1173 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        self.split = split
        assert split in {'train', 'val'}

        root_path = '/content/drive/MyDrive/gpt_2_wiki2M/data/'
        files = os.listdir(root_path)
        for f in files:
          if split in f:
            req_file = os.path.join(root_path, f)

        with open(req_file, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'For the split {split}')
        print(f"loaded for {split} {len(self.tokens)} tokens")
        print(f"for 1 epoch for {split} we will need {len(self.tokens) // (self.B * self.T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+(B * T) + 1]
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))

        #loading the next position
        self.current_position += B*T
        if (self.current_position + (B * T + 1) > len(self.tokens)):
            self.current_position = 0
        return x, y 

    def reset(self):
        self.current_position = 0
    
def count_parameters(model):
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Trainable parameters (requires_grad = True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def estimate_val_loss(model, val_loader, device, val_loss_steps=20):
    model.eval()
    val_loader.reset()

    val_loss_accum = 0.0
    with torch.no_grad():
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            val_loss_accum += loss.detach()

    val_loss = val_loss_accum / val_loss_steps
    return val_loss.item()


def main():
  model = GPT(GPTConfig(vocab_size=50304))
  model.to(device)
  model = torch.compile(model)
  print('model is on the device:', next(model.parameters()).device)

  if hasattr(model, '_orig_mod'):
      print("Model is compiled.")
  else:
      print("Model is a standard eager-mode module.")
  total, trainable = count_parameters(model)
  print(f"Total Parameters: {total:,}")
  print(f"Trainable Parameters: {trainable:,}")

  B = 4 #batch_size
  T = 512 #timestep 
  batch_size_tokens = 8192 # 2 ** 12
  grad_accum_steps = batch_size_tokens // (B*T)

  train_loader = DataLoaderLite(B, T, 'train')
  val_loader = DataLoaderLite(B, T, 'val')
  print('BatchSize:', B , 'Sequence_Length:', T, 'batch_size_tokens:', batch_size_tokens, 'grad_accum_steps', grad_accum_steps)

  #optimizer 
  optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)


  log_dir = '/content/drive/MyDrive/gpt_2_wiki2M/log/'
  log_file = os.path.join(log_dir, "log_1173.txt")
  with open(log_file, "w") as f:
      pass 

  print('-------------------- Training started --------------------')
  times = []
  tokens_per_sec_list = []
  eval_interval = 20
  val_loss_steps = 20

  print(f'The model will train for maximum of {max_steps}')
  for i in range(max_steps):
      last_step = (i == max_steps - 1)
      loss_accum = 0.0
      if device == "cuda":
          torch.cuda.synchronize()
      t0 = time.time()

      #perform validation every once in a while
      # validation every once in a while
      if i % eval_interval == 0 or last_step:
          val_loss = estimate_val_loss(model, val_loader, device, val_loss_steps=val_loss_steps)
          print(f"train step {i} validation loss: {val_loss:.4f}")
          with open(log_file, "a") as f:
              f.write(f"{i} val {val_loss:.6f}\n")


      model.train()
      #do one step of the optimization
      optimizer.zero_grad(set_to_none=True)
      for microsteps in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)  
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum+=loss.detach()
        loss.backward()

      #Clipping the global norm of the gradient to 1.0 as per gpt 3 paper
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      #print(f"For iteration {i}, the loss is {loss.item()}")

      # Learning rate calculation and doing optimzation step
      lr = get_lr(i)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
      optimizer.step()


      if device == "cuda":
          torch.cuda.synchronize()
      #torch.cuda.synchronize() # we are waiting for all the operations on gpu to finish
      t1 = time.time()
      dt = (t1 - t0)*1000
      tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
      if i >= 5:   # skip warmup
          times.append(dt)
          tokens_per_sec_list.append(tokens_per_sec)
      if(i<=10 or i>=(max_steps - 11)):    
        print(f"For iteration {i}| loss: {loss_accum} | lr:{lr:.4e} | max|logits|:{logits.detach().abs().max().item()} | norm: {norm:.4f} | dt: {dt:.2f}ms tokens_per_sec: {tokens_per_sec:.2f}")

      with open(log_file, "a") as f:
        f.write(f"{i} train {loss_accum.item():.6f}\n")
        
      save_interval = 150

      if i % save_interval == 0 and i > 0:
          checkpoint_path = os.path.join(log_dir, f"model_{i:05d}.pt")
          raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
          checkpoint = {
              'model_state_dict': raw_model.state_dict(),
              'config': raw_model.config.__dict__,
              'step': i,
              'optimizer_state_dict': optimizer.state_dict(),
          }
          torch.save(checkpoint, checkpoint_path)
          print(f"checkpoint saved at step {i}")

  times = np.array(times)
  print("median ms:", np.median(times) )
  print("mean ms:", np.mean(times))
  print("p95 ms:", np.percentile(times, 95))
  print("median tokens_per_sec:", np.median(tokens_per_sec_list) )
  print("mean tokens_per_sec:", np.mean(tokens_per_sec_list))
  print("p95 tokens_per_sec:", np.percentile(tokens_per_sec_list, 95))
 

if __name__ == "__main__":
    main()