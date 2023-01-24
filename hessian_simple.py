import torch
import matplotlib.pyplot as plt
import os
import re

from loss_landscape.utils import AttrDict, sorted_nicely
from loss_landscape.plot_hessian_eigen import plot_hessian_eigen_models, get_loss
from src.modeling import TrainableTransformer

lightning_module_class = TrainableTransformer

#selected_epochs, min_eig, max_eig = [0, 1, 2, 3, 4], [2, 10, 50, 100, 101], [5, 5, 15, 15, 40]
#phases = { 'pre_memo_epoch' : 1, 'memo_epoch' : 2, 'pre_comp_epoch' : 3,  'comp_epoch' : 4}

def plot_heigens(selected_epochs, min_eig, max_eig, save_file_name="", plot_ratio=True, phases=None, axs=None) :
    # plotting the given graph
    if axs is None :
        L, C = 1, (3 if plot_ratio else 2)
        figsize=(5*C, 4*L)
        fig, axs = plt.subplots(L, C, sharex=False, sharey=False, figsize = figsize)

    color = '#000000'
    color = None
    axs[0].plot(selected_epochs, min_eig, 
    #        marker = "+", markersize = 15, color = "red",
            label = "λ_min",
            color = color
    )
    axs[1].plot(selected_epochs, max_eig, 
    #        marker = ".", markersize = 15, color = "green",
            label = "λ_max",
            color = color
    )
    
    if plot_ratio :
            axs[2].plot(
            #axs[2].semilogy(    
                selected_epochs, 
                [a/b for a, b in zip(max_eig, min_eig)],
                #[abs(a/b) for a, b in zip(max_eig, min_eig)], 
            #    marker = "o", markersize = 15, color = "black",
                label = "λ_max/λ_min",
                color = color
            )

    # plot with grid
    #plt.grid(True)
    for ax in axs : ax.grid(True)

    if phases is not None :
        #colors = ["b", 'r', 'g', 'y']
        #colors = ['#440154', '#30678d', '#35b778', '#fde724']
        colors = ['#440154', '#3b518a', '#218e8c', '#57c665']
        colors.reverse()
        labels = {
            'pre_memo_epoch' : 'train_acc~5%', 
            'pre_comp_epoch' : 'val_acc~5%', 
            'memo_epoch' : 'train_acc~99%', 
            'comp_epoch' : 'val_acc~99%'
        }
        assert set(phases.keys()).issubset(set(labels.keys()))
        for i, k in enumerate(phases.keys()) :
            for ax in axs :
                ax.axvline(x = phases[k], color = colors[i], label = labels[k])

    for ax, ylabel in zip(axs, ["λ_min", "λ_max", "λ_min/λ_max"]) :
        ax.set(xlabel='epochs', 
              #ylabel=ylabel
        )
        ax.legend()

    if (axs is not None) and save_file_name :
        plt.savefig(save_file_name  + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_file_name  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
        
    # show the plot
    plt.show()


#####################################################

# step 1 : train the model

"""
max_epochs=800
use_wandb=False

```bash
train.sh 90 +
```
"""

# step 2 : load the checkpoints, the data and the params

train_data_pct=90
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
for k in [2, 100, 200] : good_epochs.extend([k]) # good_epochs.extend([k-1, k, k+1])
# phases
for p in phases : good_epochs.extend([p-1, p, p+1])
# slingshot
#for k in [450, 578, 765] : good_epochs.extend([k-1, k, k+1])
# end
for k in [L-6, L-3] : good_epochs.extend([k]) # good_epochs.extend([k-1, k, k+1])
####
print(len(good_epochs), good_epochs)

selected_epochs = good_epochs + []
# selected_epochs = list(range(0, 500+1, 10)) + list(range(150, L, 10))
print(selected_epochs)

selected_epochs = sorted(list(dict.fromkeys(selected_epochs)))
tmp_model_files = [model_files[e] for e in selected_epochs]
len(tmp_model_files)

# step 3 : define the parameters for the plot

args = AttrDict({ 
    'cuda' : True, # use cuda
    'threads' : 2, # 'number of threads'
    'ngpu' : 1, # 'number of GPUs to use for each rank, useful for data parallel evaluation

    # direction parameters
    'dir_type' : 'weights', #'direction type: weights | states (including BN\'s running_mean/var)'
    "tol" : 1e-2,
    "clear_freq" : 1000000
})

# step 4 : data

if True :
    dataloader = data_module.train_dataloader()
    data_size = len(data_module.train_dataset)
else :
    dataloader = data_module.val_dataloader()
    data_size = len(data_module.val_dataset)


# step 4 : compute

min_eig, max_eig = plot_hessian_eigen_models(
    args, 
    tmp_model_files,
    lightning_module_class, 
    dataloader, 
    data_size, 
    get_loss,
    LOG_PATH = PLOTS_PATH
) 

torch.save([selected_epochs, min_eig, max_eig], os.path.join(PLOTS_PATH, "eigens.pth"))

# step 5 : plots

phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
phases = {k : states[k] for k in phases_k}

save_file_name = os.path.join(PLOTS_PATH, "hessian")
plot_heigens(selected_epochs, min_eig, max_eig, save_file_name=save_file_name, plot_ratio=True, phases=phases, axs=None)