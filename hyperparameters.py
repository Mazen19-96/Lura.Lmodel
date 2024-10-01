#hyperparameters

def get_parameters():
    batch_size = 64 
    block_size = 256
    max_iters = 10
    eval_interval = 500
    learning_rate = 3e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    eval_iters = 200
    n_embd = 384 # or d_model = n_heads * d_head
    n_head = 6 
    n_layer = 6
    dropout = 0.2
    return batch_size,block_size,max_iters,eval_interval,learning_rate,device,eval_iters,n_embd,n_head,n_layer,dropout