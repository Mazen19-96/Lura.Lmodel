import torch
import hyperparameters
from hyperparameters import *

torch.manual_seed(1337)

#here we imoort our dataset and read it in to inspect it
with open("input.txt",'r',encoding='utf-8') as f:
    text=f.read()


#here are all the unique characters that occur in this text
chars=sorted(list(set(text)))
vocab_size=len(chars)


#toknizeing:creat a mapping from characters to integers
stai={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stai[c] for c in s]
decode=lambda l: ''.join([itos [i] for i in l])

data=torch.tensor(encode(text),dtype=torch.long)
#here we just split uo the data into train and valdation sets
n=int(0.9*len(data))# first 90% will be train , rest val
train_data=data[:n]
val_data=data[n:]
def get_batch(split):
    # generate a small batch of data of input x and target y
    data=train_data if split == 'train' else val_data
    ix=torch.randint(len(data) - block_size , (batch_size,))
    x= torch.stack([data[i:i+block_size] for i in ix])
    y= torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x , y
