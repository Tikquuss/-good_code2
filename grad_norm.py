import torch
import os
import re

from loss_landscape.utils import AttrDict, sorted_nicely
from loss_landscape.plot_hessian_eigen import get_loss
from loss_landscape.grad_norm import compute_grad_norm, plot_grad_norm
from src.modeling import TrainableTransformer

lightning_module_class = TrainableTransformer


# step 1 : train the model

"""
max_epochs=800
use_wandb=False

```bash
train.sh 90 +
```
"""

# step 2 : load the checkpoints, the data and the params

train_data_pct=80
math_operator="+"

LOG_PATH="D:/Canada/MILA/ppsp/loss_landscape/all_logs"
# LOG_PATH="E:/all_logs"

logdir = LOG_PATH + f"/{math_operator}/tdp={train_data_pct}-wd=1-d=0.0-opt=adamw-mlr=0.001-mo{math_operator}"
PLOTS_PATH = LOG_PATH + "/results/" + f"{math_operator}_tdp={train_data_pct}"
os.makedirs(PLOTS_PATH, exist_ok=True)
# Modification in  model_loader.py
"""
```python
LOG_PATH="D:/Canada/MILA/ppsp/loss_landscape/all_logs"
train_data_pct=30
math_operator="+"
logdir = LOG_PATH + f"/{math_operator}/tdp={train_data_pct}-wd=1-d=0.0-opt=adamw-mlr=0.001-mo{math_operator}"
hparams = torch.load(logdir + "/hparams.pt")
hparams.use_wandb = False
```
"""

pretrained_folder = logdir + "/checkpoints"

#pattern = '^epoch_[0-9]+.ckpt$'
pattern = '^epoch=[0-9]+-val_accuracy=[0-9]+\.[0-9]+.ckpt$'

model_files = os.listdir(pretrained_folder)
model_files = [f for f in model_files if re.match(pattern, f)]
model_files = sorted_nicely(model_files)
model_files = ["init.ckpt"] + model_files
model_files = [pretrained_folder + "/" + f for f in model_files]

L = len(model_files)

hparams = torch.load(logdir + "/hparams.pt")
data_module = torch.load(logdir+"/data.pt")
states = torch.load(logdir+"/states.pt")

# step :

phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
phases = [states[k] for k in phases_k]
print(phases)


good_epochs = []
# start
for k in [2, 100] : good_epochs.extend([k]) # good_epochs.extend([k-1, k, k+1])
# phases
for p in phases : good_epochs.extend([p-1, p, p+1])
# slingshot
#for k in [450, 578, 765] : good_epochs.extend([k-1, k, k+1])
# end
for k in [L-6, L-3] : good_epochs.extend([k]) # good_epochs.extend([k-1, k, k+1])
####
print(len(good_epochs), good_epochs)

selected_epochs = good_epochs + []
#selected_epochs += list(range(0, 500+1, 10)) + list(range(150, L, 10))
selected_epochs += list(range(0, L, 50))

selected_epochs = sorted(list(dict.fromkeys(selected_epochs)))
print(len(selected_epochs), selected_epochs)

# step 3 : define the parameters for the plot

args = AttrDict({ 
    'cuda' : True, # use cuda
    'threads' : 2, # 'number of threads'
    'ngpu' : 1, # 'number of GPUs to use for each rank, useful for data parallel evaluation

    # direction parameters
    'dir_type' : 'weights', #'direction type: weights | states (including BN\'s running_mean/var)'
    "clear_freq" : 1000000
})

# step 4 : data

if True :
    dataloader = data_module.train_dataloader()
    data_size = len(data_module.train_dataset)
else :
    dataloader = data_module.val_dataloader()
    data_size = len(data_module.val_dataset)

tmp_model_files = [model_files[e] for e in selected_epochs]
grad_norm_list = compute_grad_norm(args, tmp_model_files, lightning_module_class, dataloader, data_size, get_loss, LOG_PATH = PLOTS_PATH)

phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
phases = {k : states[k] for k in phases_k}

save_file_name = os.path.join(PLOTS_PATH, "grad_norm")

plot_grad_norm(selected_epochs, grad_norm_list, save_file_name=save_file_name, phases=phases, ax=None)