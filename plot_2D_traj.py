import os 
import re
import shutil
import torch
import h5py
import sys

from loss_landscape.utils import sorted_nicely
from loss_landscape.plot_trajectory import plot_trajectory
from loss_landscape.utils import AttrDict, images_to_vidoe, sorted_nicely
from src.modeling import TrainableTransformer
from loss_landscape.plot_utils import clean_h5_folder
from loss_landscape.plot_surface import plot_surface, plot_surface_list

lightning_module_class = TrainableTransformer

math_operator="+"
#math_operator="s5"
train_data_pct=30

if math_operator=="+": LOG_PATH="D:/Canada/MILA/ppsp/loss_landscape/all_logs"
elif math_operator=="s5" : LOG_PATH="F:/all_logs"
logdir = LOG_PATH + f"/{math_operator}/tdp={train_data_pct}-wd=1-d=0.0-opt=adamw-mlr=0.001-mo{math_operator}"
PLOTS_PATH = LOG_PATH + "/results/" + f"{math_operator}_tdp={train_data_pct}"
os.makedirs(PLOTS_PATH, exist_ok=True)
##
pretrained_folder = logdir + "/checkpoints"

#pattern = '^epoch_[0-9]+.ckpt$'
pattern = '^epoch=[0-9]+-val_accuracy=[0-9]+\.[0-9]+.ckpt$'

model_files = os.listdir(pretrained_folder)
model_files = [f for f in model_files if re.match(pattern, f)]
model_files = sorted_nicely(model_files)
model_files = ["init.ckpt"] + model_files
model_files = [pretrained_folder + "/" + f for f in model_files]

L = len(model_files)
print(L)

##
hparams = torch.load(logdir+"/hparams.pt")
data_module = torch.load(logdir+"/data.pt")
states = torch.load(logdir+"/states.pt")
print(states)

##
#clean_h5_folder(pretrained_folder, model_files)

##############################################################
do_projection = False
if do_projection :
    args_proj = AttrDict({ 
        'model_folder' : pretrained_folder, # 'folders for models to be projected'
        'dir_type' : 'weights', #"""direction type: weights (all weights except bias and BN paras) states (include BN.running_mean/var)""")
        'ignore' : '', # 'ignore bias and BN paras: biasbn (no bias or bn)')'
        'save_epoch' : 1, # 'save models every few epochs')

        'dir_file' : '', #'load the direction file for projection')
    })

    #selected_model_files = model_files
    selected_epochs = list(range(0, L, 10))
    selected_model_files = [ model_files[t] for t in selected_epochs]
    len(selected_model_files)

    args_proj.model_file = model_files[-1]
    proj_file_, dir_file_ = plot_trajectory(args_proj, selected_model_files, lightning_module_class)
    print(proj_file_)
    print(dir_file_)
else :
    proj_file_ = f"{logdir}/checkpoints/PCA_from_pointweights_save_epoch=1/directions.h5_proj_cos.h5"
    dir_file_ = f"{logdir}/checkpoints/PCA_from_pointweights_save_epoch=1/directions.h5"

f = h5py.File(proj_file_, 'r')
x = f['proj_xcoord'][:]
y = f['proj_ycoord'][:]
print(min(x), max(x))
print(min(y), max(y))
f.close()

sys.exit("PROJECTION")

##############################################################

args = AttrDict({ 
    
    'mpi' : False, # use mpi
    'cuda' : True, # use cuda
    'threads' : 2, # 'number of threads'
    'ngpu' : 1, # 'number of GPUs to use for each rank, useful for data parallel evaluation

    # data parameters

    'raw_data' :False, # 'no data preprocessing'
    'data_split' : 1, #'the number of splits for the dataloader'
    'split_idx' : 0, # 'the index of data splits for the dataloader'

    # model parameters
    
    # parser.add_argument('--model', default='resnet56', help='model name')
    # parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    #'model_file' : model_files[0], # path to the trained model file
    'model_file' : "", # path to the trained model file
    #'model_file2' : model_files[-1], # use (model_file2 - model_file) as the xdirection
    'model_file2' : "", # use (model_file2 - model_file) as the xdirection
    'model_file3' : "", # use (model_file3 - model_file) as the ydirection
    #'loss_name' : 'crossentropy', # help='loss functions: crossentropy | mse')

    # direction parameters

    'dir_file' : '',  # 'specify the name of direction file, or the path to an eisting direction file
    'dir_type' : 'weights', #'direction type: weights | states (including BN\'s running_mean/var)'
    'x' : '-1:1:51', #'A string with format xmin:x_max:xnum'
    'y' : None, #'A string with format ymin:ymax:ynum'
    #'y' : '-1:1:51', #'A string with format ymin:ymax:ynum'
    'xnorm' : '', # 'direction normalization: filter | layer | weight'
    'ynorm' : '', # 'direction normalization: filter | layer | weight'
    'xignore' : '', #'ignore bias and BN parameters: biasbn'
    'yignore' : '', #'ignore bias and BN parameters: biasbn'
    'same_dir' : False, # 'use the same random direction for both x-axis and y-axis'
    'idx' : 0, # 'the index for the repeatness experiment')
    'surf_file' : '', # 'customize the name of surface file, could be an existing file.'

    # plot parameters

    'proj_file' : '', # 'the .h5 file contains projected optimization trajectory.'
    'loss_max' : None, # 'Maximum value to show in 1D plot'
    'acc_max' :None, # 'ymax value (accuracy)')
    'vmax' : 10, # 'Maximum value to map'
    'vmin' : 0.1, # 'Miminum value to map'
    'vlevel' : 0.5, # 'plot contours every vlevel'
    'show' : True, # 'show plotted figures'
    'log' : False, # 'use log scale for loss values'
    'plot' : True, # 'plot figures after computation'
})

args.dir_type='weights'
#args.dir_type='states'
args.xnorm="filter"
#args.xnorm="layer"
#args.xnorm="weight"
args.threads = 4

args.dir_file = dir_file_
args.proj_file = proj_file_

#####

args.model_file = model_files[-1]
#args.model_file = ""
args.model_file2 = ""

#args.x='-1:1:51'
#args.y='-1:1:51'

args.x='-1:1:5'
args.y='-1:1:6'

# args.x='-2:2:25'
# args.y='-2:2:25'

# args.x='-5:5:50'
# args.y='-5:5:50'

# args.x='-1:160:100'
# args.y='-20:6:27'

# dir_file5, surf_file5 = plot_surface(
#         args, lightning_module_class, metrics = ['val_loss', 'val_accuracy'],
#         train_dataloader = data_module.train_dataloader(), 
#         test_dataloader = data_module.val_dataloader(),
#         save_to = os.path.join(PLOTS_PATH, "3D_trajectory")
#     )

# print(dir_file5)
# print(surf_file5)