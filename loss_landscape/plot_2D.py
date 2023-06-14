"""
    2D plotting funtions
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import os
import seaborn as sns
from matplotlib.ticker import LinearLocator


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False, save_to = ""):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
        # update
        Z = Z.T
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
        # update
        Z = Z.T
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    if save_to : filename = os.path.join(save_to, surf_name + '_2dcontour')
    else : filename=surf_file + '_' + surf_name + '_2dcontour'
    fig.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(filename + '.png', dpi=300, bbox_inches='tight')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    if save_to : filename = os.path.join(save_to, surf_name + '_2dcontourf')
    else : filename = surf_file + '_' + surf_name + '_2dcontourf' 
    fig.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(filename + '.png', dpi=300, bbox_inches='tight')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    if save_to : filename = os.path.join(save_to, surf_name + '_2dheat')
    else : filename = surf_file + '_' + surf_name + '_2dheat' 
    sns_plot.get_figure().savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    sns_plot.get_figure().savefig(filename + '.png', dpi=300, bbox_inches='tight')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ####### https://matplotlib.org/stable/gallery/mplot3d/surface3d.html #######  
    R, C = 1, 1
    #figsize=(C*15, R*10)
    figsize=(C*6, R*4)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #######
    if save_to : filename = os.path.join(save_to, surf_name + '_3dsurface')
    else : filename = surf_file + '_' + surf_name + '_3dsurface' 
    fig.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(filename + '.png', dpi=300, bbox_inches='tight')

    f.close()
    if show: plt.show()


def plot_trajectory(proj_file, dir_file, show=False, save_to = ""):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    assert exists(proj_file), 'Projection file does not exist.'
    f = h5py.File(proj_file, 'r')
    fig = plt.figure()
    try :
        plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')
    except ValueError: #2 indexing arguments for 1 dimensions
        plt.plot(f['proj_xcoord'][:], f['proj_ycoord'][:], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')
    f.close()

    if exists(dir_file):
        f2 = h5py.File(dir_file,'r')
        if 'explained_variance_ratio_' in f2.keys():
            ratio_x = f2['explained_variance_ratio_'][0]
            ratio_y = f2['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        f2.close()

    if save_to : filename = os.path.join(save_to, proj_file)
    else : filename = proj_file + "" 
    fig.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    if show: plt.show()


def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False, save_to = ""):
    """2D contour + trajectory"""

    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    # update
    Z = Z.T

    fig = plt.figure()
    #CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
    #CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

    CS1 = plt.contour(X, Y, Z)
    CS2 = plt.contour(X, Y, Z)

    # plot trajectories
    pf = h5py.File(proj_file, 'r')
    plt.plot(pf['proj_xcoord'][:], pf['proj_ycoord'][:], marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    df = h5py.File(dir_file,'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    #df.close()
    plt.clabel(CS1, inline=1, fontsize=6)
    plt.clabel(CS2, inline=1, fontsize=6)
    if save_to : filename = os.path.join(save_to, surf_name + '_2dcontour_proj')
    else : filename = proj_file + '_' + surf_name + '_2dcontour_proj'
    fig.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    
    df.close()
    pf.close()
    f.close()
    if show: plt.show()


def plot_2d_eig_ratio(surf_file, val_1='min_eig', val_2='max_eig', show=False, save_to = ""):
    """ Plot the heatmap of eigenvalue ratios, i.e., |min_eig/max_eig| of hessian """

    print('------------------------------------------------------------------')
    print('plot_2d_eig_ratio')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])

    # Plot 2D heatmaps with color bar using seaborn
    abs_ratio = np.absolute(np.divide(Z1, Z2))
    print(abs_ratio)

    fig = plt.figure()
    sns_plot = sns.heatmap(abs_ratio, cmap='viridis', vmin=0, vmax=.5, cbar=True,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    if save_to : filename = os.path.join(save_to, val_1 + '_' + val_2 + '_abs_ratio_heat_sns')
    else : filename = surf_file + '_' + val_1 + '_' + val_2 + '_abs_ratio_heat_sns'
    sns_plot.get_figure().savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    sns_plot.get_figure().savefig(filename + '.png', dpi=300, bbox_inches='tight')

    # Plot 2D heatmaps with color bar using seaborn
    ratio = np.divide(Z1, Z2)
    print(ratio)
    fig = plt.figure()
    sns_plot = sns.heatmap(ratio, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    if save_to : filename = os.path.join(save_to,  val_1 + '_' + val_2 + '_ratio_heat_sns')
    else : filename = surf_file + '_' +  val_1 + '_' + val_2 + '_ratio_heat_sns'
    sns_plot.get_figure().savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    sns_plot.get_figure().savefig(filename + '.png', dpi=300, bbox_inches='tight')
    f.close()
    if show: plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D loss surface')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
    parser.add_argument('--dir_file', default='', help='The h5 file that contains directions')
    parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--zlim', default=10, type=float, help='Maximum loss value to show')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--save_to', default="", type=str, help='')

    args = parser.parse_args()

    if exists(args.surf_file) and exists(args.proj_file) and exists(args.dir_file):
        plot_contour_trajectory(args.surf_file, args.dir_file, args.proj_file,
                                args.surf_name, args.vmin, args.vmax, args.vlevel, args.show, save_to=args.save_to)
    elif exists(args.proj_file) and exists(args.dir_file):
        plot_trajectory(args.proj_file, args.dir_file, args.show, save_to=args.save_to)
    elif exists(args.surf_file):
        plot_2d_contour(args.surf_file, args.surf_name, args.vmin, args.vmax, args.vlevel, args.show, save_to=args.save_to)
