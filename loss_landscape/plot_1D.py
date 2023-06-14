"""
    1D plotting routines
"""

import imp
from matplotlib import pyplot as pp
import h5py
import argparse
import numpy as np
import math

from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

def get_twin_axis(color_1="k", color_2="k") :

    color='black' # major : 'k' 'gray' 'black' ...
    color_minor='gray' # minor
    linestyle="-"
    linestyle_minor='--'
    linewidth=0.3 # major
    linewidth_minor=0.2 # minor
    alpha=0.4
    alpha_minor=0.3

    R, C = 1, 1
    #figsize=(C*15, R*10)
    figsize=(C*6, R*4)
    fig, ax1 = pp.subplots(figsize=figsize)

    # 
    ############################ solution 1 ############################
    # ax1.grid(linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)
    # ax2 = ax1.twinx()
    # ax2.grid(linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)
    
    ############################ solution 2 ############################
    # ax1.grid(axis='x', linestyle=linestyle, which='major', color=color, linewidth=linewidth, alpha=alpha)   # x major
    # ax1.grid(axis='y', linestyle=linestyle, color=color_1, linewidth=linewidth, alpha=alpha)
    # pp.minorticks_on() 
    # ax2 = ax1.twinx()
    # ax2.grid(axis='y', linestyle=linestyle, color=color_2, linewidth=linewidth, alpha=alpha)
    # pp.minorticks_on() 

    # 
    ############################ solution 3 ############################
    ax1.grid(axis='x', linestyle=linestyle, which='major', color=color, linewidth=linewidth, alpha=alpha) 
    ax1.grid(axis='x', linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)  
    ax1.grid(axis='y', linestyle=linestyle, which='major', color=color_1, linewidth=linewidth, alpha=alpha)
    ax1.grid(axis='y', linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    pp.minorticks_on() 
    ax2 = ax1.twinx()
    ax2.grid(axis='y', linestyle=linestyle, which='major', color=color_2, linewidth=linewidth, alpha=alpha)
    ax2.grid(axis='y', linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    pp.minorticks_on() 
    ############################ ######## # ############################
    
    return fig, ax1, ax2

def add_legend(ax1, ax2, color_loss, color_acc, train_label, val_label):
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
    #bbox_to_anchor=(0., 1.02, 1., .102) # top
    #bbox_to_anchor=(0.5, 1.09) # top
    bbox_to_anchor=(0.5, 1.05) # top, on the line
    ##
    legend_elements_train = (Line2D([0], [0], linestyle='-', color=color_loss), Line2D([0], [0], linestyle='-', color=color_acc))
    legend_elements_val = (Line2D([0], [0], linestyle='--', color=color_loss), Line2D([0], [0], linestyle='--', color=color_acc))
    ################ 1 ################
    # ax2.legend(handles=legend_elements_train, labels=['', train_label], loc='upper right', ncol=2, bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True,
    #            # labelspacing=1, borderpad=0.4
    #            )
    # ##
    # leg = Legend(ax2, legend_elements_val, ['', val_label], loc='upper left', ncol=2, bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True, 
    #             #frameon=False, title=''
    #             )
    # ax2.add_artist(leg)
    ################ 1 ################
    ################ 2 ################
    # legend_elements_train = [legend_elements_train]
    # legend_elements_val = [legend_elements_val]
    # ##
    # ax2.legend(legend_elements_train, [train_label], loc='upper right',  ncol=2, bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True,
    #         handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5)
    # ##
    # leg = Legend(ax2, legend_elements_val, [val_label], loc='upper left', ncol=2, bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True, 
    #             handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5)
    # ax2.add_artist(leg)
    ################ 2 ################
    ################ 3 ################
    ax2.legend([legend_elements_train, legend_elements_val], [train_label, val_label], 
               #loc='upper center', 
            #ncol=2, #bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True,
            handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5,
            fontsize='large',
            )
    ################ 3 ################

def plot_1d_loss_err(surf_file, xmin=-1.0, xmax=1.0, loss_max=None, acc_max = None, log=False, show=False, save_to = None):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    y_loss_min, y_loss_max = min(train_loss), loss_max if loss_max else max(train_loss)
    y_acc_min, y_acc_max = min(train_acc), acc_max if acc_max else max(train_acc)

    save_to = save_to if save_to else surf_file

    ############################ update ############################
    # https://matplotlib.org/stable/tutorials/colors/colors.html
    color_loss = 'tab:blue' # #1f77b4
    color_acc = 'tab:red' # #d62728
    train_label = 'train'
    val_label = "validation" # 'val' "validation" "test"
    ############################ update ############################

    # loss and accuracy map
    fig, ax1, ax2 = get_twin_axis(color_loss, color_acc)

    if log:
        tr_loss, = ax1.semilogy(x, train_loss, '-', color = color_loss, label=train_label, linewidth=1) # Training loss
    else:
        tr_loss, = ax1.plot(x, train_loss, '-', color = color_loss, label=train_label, linewidth=1) # Training loss
    tr_acc, = ax2.plot(x, train_acc, '-', color = color_acc, label=train_label, linewidth=1) # Training accuracy

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        y_loss_min, y_loss_max = min(y_loss_min, min(test_loss)), max(y_loss_max, max(test_loss))
        y_acc_min, y_acc_max = min(y_acc_min, min(test_acc)), max(y_acc_max, max(test_acc))
        if log:
            te_loss, = ax1.semilogy(x, test_loss, '--', color = color_loss, label=val_label, linewidth=1) # Test loss
        else:
            te_loss, = ax1.plot(x, test_loss, '--', color = color_loss, label=val_label, linewidth=1) # Test loss
        te_acc, = ax2.plot(x, test_acc, '--', color = color_acc, label=val_label, linewidth=1) # Test accuracy

    # err_loss = y_loss_max - y_loss_min
    # y_loss_min -= err_loss / y_loss_max
    # y_loss_max += err_loss / y_loss_max

    # err_acc = y_acc_max - y_acc_min
    # y_acc_min -= err_acc / 100
    # y_acc_max += err_acc / 100

    #if log: y_loss_min, y_loss_max = math.log(y_loss_min, math.e), math.log(y_loss_max, math.e)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color=color_loss, fontsize='xx-large')
    ax1.tick_params('y', colors=color_loss, labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    #ax1.set_ylim(0, loss_max)
    ax1.set_ylim(y_loss_min, y_loss_max)
    ax2.set_ylabel('Accuracy', color=color_acc, fontsize='xx-large')
    ax2.tick_params('y', colors=color_acc, labelsize='x-large')
    #ax2.set_ylim(0, acc_max)
    ax2.set_ylim(y_acc_min, y_acc_max)

    ############################ update ############################
    add_legend(ax1, ax2, color_loss, color_acc, train_label, val_label)
    ############################ update ############################

    filename = save_to + '_1d_loss_acc' + ('_log' if log else '')
    pp.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    # # train_loss curve
    # fig = pp.figure()
    # if log:
    #     pp.semilogy(x, train_loss)
    # else:
    #     pp.plot(x, train_loss)
    # pp.ylabel('Training Loss', fontsize='xx-large')
    # pp.xlim(xmin, xmax)
    # #pp.ylim(0, loss_max)
    # pp.ylim(y_loss_min, y_loss_max)
    # filename = save_to + '_1d_train_loss' + ('_log' if log else '')
    # pp.savefig(filename + '.pdf',
    #             dpi=300, bbox_inches='tight', format='pdf')
    # fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    # # train_err curve
    # fig = pp.figure()
    # tmp = 100.0 - train_acc
    # pp.plot(x, tmp)
    # pp.xlim(xmin, xmax)
    # #pp.ylim(0, acc_max)
    # pp.ylim(min(tmp), max(tmp))
    # pp.ylabel('Training Error', fontsize='xx-large')
    # filename = save_to + '_1d_train_err'
    # pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    # fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show: pp.show()
    f.close()

def plot_1d_loss_err_old(surf_file, xmin=-1.0, xmax=1.0, loss_max=None, acc_max = None, log=False, show=False, save_to = None):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print("train_loss")
    print(train_loss)
    print("train_acc")
    print(train_acc)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    y_loss_min, y_loss_max = min(train_loss), loss_max if loss_max else max(train_loss)
    y_acc_min, y_acc_max = min(train_acc), acc_max if acc_max else max(train_acc)

    save_to = save_to if save_to else surf_file

    # loss and accuracy map
    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()
    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='train', linewidth=1) # Training loss
    else:
        tr_loss, = ax1.plot(x, train_loss, 'b-', label='train', linewidth=1) # Training loss
    tr_acc, = ax2.plot(x, train_acc, 'r-', label='train', linewidth=1) # Training accuracy

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        y_loss_min, y_loss_max = min(y_loss_min, min(test_loss)), max(y_loss_max, max(test_loss))
        y_acc_min, y_acc_max = min(y_acc_min, min(test_acc)), max(y_acc_max, max(test_acc))
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='validation', linewidth=1) # Test loss
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='validation', linewidth=1) # Test loss
        te_acc, = ax2.plot(x, test_acc, 'r--', label='validation', linewidth=1) # Test accuracy

    # err_loss = y_loss_max - y_loss_min
    # y_loss_min -= err_loss / y_loss_max
    # y_loss_max += err_loss / y_loss_max

    # err_acc = y_acc_max - y_acc_min
    # y_acc_min -= err_acc / 100
    # y_acc_max += err_acc / 100

    #if log: y_loss_min, y_loss_max = math.log(y_loss_min, math.e), math.log(y_loss_max, math.e)


    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    #ax1.set_ylim(0, loss_max)
    ax1.set_ylim(y_loss_min, y_loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    #ax2.set_ylim(0, acc_max)
    ax2.set_ylim(y_acc_min, y_acc_max)

    ## update ###
    ax1.legend()
    ax2.legend()
    #pp.grid(ls="--", c="k", alpha=0.4)
    pp.grid()
    ####

    filename = save_to + '_1d_loss_acc' + ('_log' if log else '')
    pp.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')



    # train_loss curve
    fig = pp.figure()
    if log:
        pp.semilogy(x, train_loss)
    else:
        pp.plot(x, train_loss)
    pp.ylabel('Training Loss', fontsize='xx-large')
    pp.xlim(xmin, xmax)
    #pp.ylim(0, loss_max)
    pp.ylim(y_loss_min, y_loss_max)
    filename = save_to + '_1d_train_loss' + ('_log' if log else '')
    pp.savefig(filename + '.pdf',
                dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    # train_err curve
    fig = pp.figure()
    tmp = 100.0 - train_acc
    pp.plot(x, tmp)
    pp.xlim(xmin, xmax)
    #pp.ylim(0, acc_max)
    pp.ylim(min(tmp), max(tmp))
    pp.ylabel('Training Error', fontsize='xx-large')
    filename = save_to + '_1d_train_err'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show: pp.show()
    f.close()


def plot_1d_loss_err_repeat(prefix, idx_min=1, idx_max=10, xmin=-1.0, xmax=1.0,
                            loss_max=None, acc_max = 100, show=False, save_to = None):
    """
        Plotting multiple 1D loss surface with different directions in one figure.
    """

    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()

    y_loss_min, y_loss_max = 0.0, 1.0e9
    y_acc_min, y_acc_max = 0.0, 100.0
    save_to = save_to if save_to else prefix

    for idx in range(idx_min, idx_max + 1):
        # The file format should be prefix_{idx}.h5
        f = h5py.File(prefix + '_' + str(idx) + '.h5','r')

        x = f['xcoordinates'][:]
        train_loss = f['train_loss'][:]
        train_acc = f['train_acc'][:]
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]

        xmin = xmin if xmin != -1.0 else min(x)
        xmax = xmax if xmax != 1.0 else max(x)

        y_loss_min, y_loss_max = min(y_loss_min, min(test_loss)), max(y_loss_max, max(test_loss))
        y_acc_min, y_acc_max = min(y_acc_min, min(test_acc)), max(y_acc_max, max(test_acc))

        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        te_loss, = ax1.plot(x, test_loss, 'b--', label='Testing loss', linewidth=1)
        tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Testing accuracy', linewidth=1)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    #ax1.set_ylim(0, loss_max)
    ax1.set_ylim(y_loss_min, y_loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    #ax2.set_ylim(0, acc_max)
    ax2.set_ylim(y_acc_min, y_acc_max)
    filename = save_to + '_1d_loss_err_repeat'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show: pp.show()


def plot_1d_eig_ratio(surf_file, xmin=-1.0, xmax=1.0, val_1='min_eig', val_2='max_eig', ymax=None, show=False, save_to = None):
    print('------------------------------------------------------------------')
    print('plot_1d_eig_ratio')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    x = f['xcoordinates'][:]

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])
    abs_ratio = np.absolute(np.divide(Z1, Z2))

    y_loss_min, y_loss_max = min(abs_ratio), ymax if ymax else max(abs_ratio)
    save_to = save_to if save_to else surf_file

    pp.plot(x, abs_ratio)
    pp.xlim(xmin, xmax)
    #pp.ylim(0, ymax)
    pp.ylim(y_loss_min, y_loss_max)
    filename = save_to + '_1d_eig_abs_ratio'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    ratio = np.divide(Z1, Z2)
    y_loss_min, y_loss_max = min(ratio), ymax if ymax else max(ratio)
    pp.plot(x, ratio)
    pp.xlim(xmin, xmax)
    #pp.ylim(0, ymax)
    pp.ylim(y_loss_min, y_loss_max)
    filename = save_to + '_1d_eig_ratio'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    f.close()
    if show: pp.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plott 1D loss and error curves')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file contains loss values')
    parser.add_argument('--log', action='store_true', default=False, help='logarithm plot')
    parser.add_argument('--xmin', default=-1, type=float, help='xmin value')
    parser.add_argument('--xmax', default=1, type=float, help='xmax value')
    parser.add_argument('--loss_max', default=5, type=float, help='ymax value (loss)')
    parser.add_argument('--acc_max', default=100, type=float, help='ymax value (accuracy)')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--prefix', default='', help='The common prefix for surface files')
    parser.add_argument('--idx_min', default=1, type=int, help='min index for the surface file')
    parser.add_argument('--idx_max', default=10, type=int, help='max index for the surface file')

    args = parser.parse_args()

    if args.prefix:
        plot_1d_loss_err_repeat(args.prefix, args.idx_min, args.idx_max,
                                args.xmin, args.xmax, args.loss_max, args.acc_max, args.show)
    else:
        plot_1d_loss_err(args.surf_file, args.xmin, args.xmax, args.loss_max, args.acc_max, args.log, args.show)
