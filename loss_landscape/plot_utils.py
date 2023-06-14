import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.text import Text

import shutil
import os

def clean_h5_folder(pretrained_folder, model_files):
    for f in os.listdir(pretrained_folder) :
        ff = pretrained_folder + "/" + f
        if ff not in model_files :
            print(f)
            try :
                try : os.remove(ff)
                except IsADirectoryError : shutil.rmtree(ff)
            except PermissionError as ex :
                print("PermissionError :", ex)

def add_legend(ax, legend_label, color_leg = 'tab:blue', font_color_leg='white', font_size=None, xtick_to_color=[], ytick_to_color=[]) :
    #bbox_to_anchor=(0., 1.02, 1., .102) # top
    #bbox_to_anchor=(0.5, 1.09) # top
    bbox_to_anchor=(0.5, 1.05) # top, on the line
    ##
    legend_elements_train = [Line2D([0], [0], linestyle='-', color=color_leg)]
    bbox = dict(boxstyle="round", ec=color_leg, fc=color_leg, alpha=0.5)
    leg=ax.legend(legend_elements_train, [legend_label], 
              #loc='upper center',  ncol=1, 
              #bbox_to_anchor=bbox_to_anchor, 
              #fancybox=True, shadow=True,
              handler_map={tuple: HandlerTuple(ndivide=None)}, #handlelength=5,
              # https://www.tutorialspoint.com/how-do-you-just-show-the-text-label-in-a-plot-legend-in-matplotlib
              handlelength=0, handletextpad=0,
              fontsize=font_size,
              # https://stackoverflow.com/a/63273370/11814682
              labelcolor='linecolor',
              # https://stackoverflow.com/a/63273370/11814682
              #facecolor=font_color_leg, labelcolor='w',
              #fontweight='bold'
              )
    #plt.setp(ax.get_legend(), fontweight=100)
    #https://stackoverflow.com/a/19863736/11814682
    frame = leg.get_frame()
    frame.set_facecolor(font_color_leg)
    frame.set_edgecolor('w')
    # https://stackoverflow.com/a/53433594/11814682
    bbox = dict(boxstyle="round", ec=color_leg, fc=color_leg, alpha=0.5)
    #plt.setp(ax.get_xticklabels(), bbox=bbox)
    # modify labels
    for tl in ax.get_xticklabels():
        txt = tl.get_text()
        if txt and (float(txt) in xtick_to_color):
            txt += ' (!)'
            #tl.set_backgroundcolor('C3')
            plt.setp(tl, bbox=bbox)
        tl.set_text(txt)
    for tl in ax.get_yticklabels():
        txt = tl.get_text()
        if txt and (float(txt) in ytick_to_color):
            txt += ' (!)'
            #tl.set_backgroundcolor('C3')
            plt.setp(tl, bbox=bbox)
        tl.set_text(txt)

    # #https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
    # textstr = legend_label
    # # these are matplotlib.patch.Patch properties
    # facecolor="none"
    # facecolor='wheat'
    # facecolor='tab:blue'
    # props = dict(boxstyle='round', facecolor=facecolor, alpha=0.9)
    # props = {'boxstyle':'round', 'facecolor':facecolor,'alpha':0.5,'edgecolor':'none','pad':1}
    # # place a text box in upper left in axes coords
    # x_text, y_text = 0.05, 0.95
    # x_text, y_text = 0.05, 0.95
    # #ax.text(x_text, y_text, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)

    # ax.text(
    #     x=x_text, y=y_text, s=textstr, 
    #     color=None, 
    #     verticalalignment= 'top', #'baseline', 
    #     horizontalalignment='left', 
    #     #multialignment=None, 
    #     #fontproperties=None, rotation=None, linespacing=None, rotation_mode=None, usetex=None, wrap=False, transform_rotates_text=False, parse_math=True,
    #     ######
    #     transform=ax.transAxes, fontsize=20,
    #     bbox=props,
    #     #ha='center', va='center'
    #     )


    # # ax.annotate(x_text,
    # #             xy=(x_text, y_text), xycoords='axes fraction',
    # #             textcoords='offset points',
    # #             size=14,
    # #             bbox=dict(boxstyle="round", facecolor=facecolor, ec="none"))


def plot_image(
    ax, fig, img, xticks, yticks, x_label, y_label, cmap=None, interpolation=None, aspect=None,
    step_xticks = 1, step_yticks = 1, xtick_to_color=[], ytick_to_color=[]
    ) :
    im = ax.matshow(
        img, 
        #origin='upper'
        origin='lower',
        cmap=cmap,
        interpolation=interpolation,
        aspect=aspect,
    )

    # color bar
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # y axis
    tmp = np.array(yticks).round(3)
    ax.set_yticks(list(range(len(tmp))))
    tmp = [s if (i%step_yticks==0 or s in ytick_to_color) else "" for i, s in enumerate(tmp)]
    ax.set_yticklabels(tmp)
    # x axis
    tmp = np.array(xticks).round(3)
    ax.set_xticks(list(range(len(tmp))))
    tmp = [s if (i%step_xticks==0 or s in xtick_to_color) else "" for i, s in enumerate(tmp)]
    ax.set_xticklabels(tmp, rotation=90)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # ticks
    font_size = 10
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # labels
    font_size = 20
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size) 