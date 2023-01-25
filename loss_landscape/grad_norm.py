import torch
from torch.autograd import Variable
import time
import socket
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model_loader import load
from .evaluation import Evaluator
from .projection import tensorlist_to_tensor

################################################################################
# Grad norm
###############################################################################

def grad_norm(net, dataloader, evaluator, data_size, use_cuda=False):
    """
    """
    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    if use_cuda:
        net.cuda()

    net.eval()
    net.zero_grad() # clears grad for every parameter in the net

    n = 0
    result = 0
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch["text"]
        targets = batch["target"]
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        loss = evaluator._get_loss(net, batch = {"text" : inputs, "target" : targets}, data_size = data_size)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)

        result += tensorlist_to_tensor(grad_f).detach().norm().item()
        n+=1
    
    return result

def compute_grad_norm(args, model_files, lightning_module_class, dataloader, data_size, get_loss, LOG_PATH = None) :
    
    # Setting the seed
    pl.seed_everything(42)

    rank = 0

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' % (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    evaluator = Evaluator(get_loss = get_loss)

    """Calculate eigen values of the hessian matrix of a given model."""
    grad_norm_list = []
    L = len(model_files)

    # Loop over all un-calculated coords
    start_time = time.time()

    clear_freq = getattr(args, "clear_freq", 10)
    logger = CSVLogger(LOG_PATH, name="grad_norm")
    try:
        #for count, f in tqdm(enumerate(model_files)):
        for count, f in enumerate(model_files):
            try:
                net = load(lightning_module_class, model_file = f)
                # data parallel with multiple GPUs on a single node
                if args.ngpu > 1: net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                
                # Compute the eign values of the hessian matrix
                compute_start = time.time()
                result = grad_norm(net, dataloader, evaluator, data_size, use_cuda=args.cuda)
                compute_time = time.time() - compute_start

                # Record the result in the local array
                grad_norm_list.append(result)
                
                if LOG_PATH :
                    try : epoch = int(f.split("-val_accuracy")[0].split("epoch=")[1])
                    except : epoch = 0
                    logger.log_metrics(metrics={"grad_norm" : result, "epoch" : epoch}, step=count+1)
                    logger.save()

                print("%d/%d  (%0.2f%%) \tgrad_norm:%8.5f \ttime:%.2f" % 
                    (count + 1, L, 100.0 * (count + 1)/L, result, compute_time))

                if count%clear_freq == 0 : 
                    #os.system('cls')
                    clear_output(wait=True)
                    
            except KeyboardInterrupt:
                pass

    except KeyboardInterrupt:
        pass
    
    total_time = time.time() - start_time
    print('Total time: %f '%total_time)

    return grad_norm_list

def plot_grad_norm(
    selected_epochs, grad_norm_list, save_file_name="", phases=None, ax=None,
    log_x = False, log_y = False
    ) :
    # plotting the given graph
    if ax is None :
        L, C = 1, 1
        figsize=(5*C, 4*L)
        fig, ax = plt.subplots(L, C, sharex=False, sharey=False, figsize = figsize)

    color = '#000000'
    color = None
    ax.plot(selected_epochs, grad_norm_list, 
            #label = "grad_norm",
            color = color
    )

    # plot with grid
    #plt.grid(True)
    ax.grid(True)

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
            ax.axvline(x = phases[k], color = colors[i], label = labels[k])

    if log_x : ax.set_xscale('log')
    if log_y : 
            ax.set_yscale('log')
            #ax.set_ylabel('log', loc = 'top', rotation="horizontal")
            #ax.yaxis.set_label_coords(0.025, 1.02)

    ax.set(xlabel='epochs', ylabel=f"gradient norm {'(log-scale)' if log_y else ''}")
    ax.legend()

    if (ax is not None) and save_file_name :
        plt.savefig(save_file_name  + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_file_name  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
        
    # show the plot
    plt.show()