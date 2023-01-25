import torch
import re
import os 
from typing import Dict, List
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from loss_landscape.cosine_sim import consine_sim_weights_states, consine_sim_vec, consine_sim_vec_from_point#, plot_cosine_sim
from loss_landscape.utils import sorted_nicely

from src.modeling import TrainableTransformer
lightning_module_class = TrainableTransformer

def plot_cosine_sim(
    angles : List[Dict], ylabel=None, phases : Dict = None, save_to = None,
    log_x = False, log_y = False
    ) :

    L, C = 1, len(angles)
    figsize = (7*C, 4*L)
    fig, axs = plt.subplots(L, C, sharex=False, sharey=False, figsize = figsize)
    if C == 1 : axs = [axs]

    for ax, angle in zip(axs, angles) :
        ax.plot(angle["epochs"], angle['angles'], label=angle["label"])
    
    if phases is not None :
        colors = ["b", 'r', 'g', 'y']
        colors = ['#440154', '#3b518a', '#218e8c', '#57c665']
        colors.reverse()
        # labels = {
        #     'pre_memo_epoch' : 'pre_memorization_epoch (train_acc~5%)', 
        #     'pre_comp_epoch' : 'pre_comprehension_epoch (val_acc~5%)', 
        #     'memo_epoch' : 'memorization_epoch (train_acc~99%)', 
        #     'comp_epoch' : 'comprehension_epoch (val_acc~99%)'
        # }
        labels = {
            'pre_memo_epoch' : 'train_acc~5%', 
            'pre_comp_epoch' : 'val_acc~5%', 
            'memo_epoch' : 'train_acc~99%', 
            'comp_epoch' : 'val_acc~99%'
        }
        assert set(phases.keys()).issubset(set(labels.keys()))
        for i, k in enumerate(phases.keys()) :
            for ax in axs : ax.axvline(x = phases[k], color = colors[i], label = labels[k])
            #axs[0].axvline(x = phases[k], color = colors[i], label = labels[k])

    #axs[0].set(ylabel=ylabel)
    for ax in axs :
        if log_x : ax.set_xscale('log')
        if log_y : 
            ax.set_yscale('log')
            ax.set_ylabel('log', loc = 'top', rotation="horizontal")
            ax.yaxis.set_label_coords(0.025, 1.02)
        
        ax.grid(True)

        #ax.set(xlabel='epochs', ylabel=ylabel)
        ax.set(xlabel='epochs')
        #ax.set_title('title')
        ax.legend()

    if save_to is not None: 
        fig.savefig(save_to+".png", dpi=300, bbox_inches='tight')
        fig.savefig(save_to+".pdf", dpi=300, bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    # step 1 : train the model

    parser = ArgumentParser(description="cosine similarity")
    parser.add_argument("--train_data_pct", type=int, help="training data fraction")
    parser.add_argument("--math_operator", type=str, default="+", help="")
    params = parser.parse_args()

    # step 2 : load the checkpoints, the data and the params

    train_data_pct=params.train_data_pct
    math_operator=params.math_operator

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

    phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
    phases = [states[k] for k in phases_k]
    print(phases)


    ## step 3 : The epochs concerning by the plots 
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
    # selected_epochs += list(range(0, 500+1, 10)) + list(range(150, L, 10))
    selected_epochs += list(range(0, L, 1))

    selected_epochs = sorted(list(dict.fromkeys(selected_epochs)))
    print(len(selected_epochs), selected_epochs)
    
    tmp_model_files = [model_files[e] for e in selected_epochs]
    len(tmp_model_files)


    # step 5 : angles, weigths level

    print("==== angles, weigths level ====")

    dir_type = 'weights'
    #dir_type = 'states'

    #ignore = 'biasbn'
    ignore = ''
    #angles1 = consine_sim_weights_states(tmp_model_files, dir_type, ignore, lightning_module_class)

    # dir_type = 'states'
    # angles11 = consine_sim_weights_states(tmp_model_files, dir_type, ignore, lightning_module_class)

    # step 6 : angles
    print("==== angles ====")
    angles2 = consine_sim_vec(tmp_model_files, lightning_module_class) 

    # step 7 : angles, from init
    print("==== angles, from init ====")
    angles3 = consine_sim_vec_from_point(model_files[0], tmp_model_files, lightning_module_class) 

    # step 8 : plot
    s = os.path.join(PLOTS_PATH, "angles.pth")
    print(f"save to {s}")
    torch.save([selected_epochs, angles2, angles3], s)

    phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
    plot_cosine_sim(
        angles = [
            #{"angles" : angles1, "label" : "cos_%s (θ_{i+1}, θ_{i})"%dir_type, "epochs" : selected_epochs[:-1]},
            {"angles" : angles2, "label" : "cos(θ_{i+1}, θ_{i})", "epochs" : selected_epochs[:-1]},
            {"angles" : angles3, "label" : "cos(theta_{i}, theta_0)", "epochs" : selected_epochs},
        ],
        ylabel="consine similarity", 
        phases = { k : states[k] for k in phases_k}, 
        save_to = os.path.join(PLOTS_PATH, f"angles_{train_data_pct}"),
        log_x = False,
        log_y = True
    )