"""
Load a pytorch lightning module checkpoint
"""

import torch
#logdir = ""
train_data_pct=85
math_operator="+"
LOG_PATH="D:/Canada/MILA/ppsp/loss_landscape/all_logs"
#LOG_PATH="E:/all_logs"
logdir = LOG_PATH + f"/{math_operator}/tdp={train_data_pct}-wd=1-d=0.0-opt=adamw-mlr=0.001-mo{math_operator}"
hparams = torch.load(logdir + "/hparams.pt")
hparams.use_wandb = False

def load(lightning_module_class, model_file):
    #return lightning_module_class.load_from_checkpoint(model_file)
    return lightning_module_class.load_from_checkpoint(hparams = hparams, checkpoint_path = model_file).float()