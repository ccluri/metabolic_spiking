import numpy as np

from gates import minisog_inf, aox_inf, rotenone_inf

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import LogLocator, NullFormatter

from palettable.cartocolors.sequential import Burg_5
from palettable.colorbrewer.qualitative import Set2_8, Set1_8, Set3_12
from palettable.colorbrewer.sequential import RdPu_5, BuPu_5, YlGn_5

style_dict = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'xtick.labelsize': 6,
    'xtick.major.size': 2,
    'xtick.major.width': 0.5,
    'xtick.major.pad': 1,
    'xtick.minor.size': 1,
    'xtick.minor.width': 0.5,
    'ytick.labelsize': 6,
    'ytick.major.size': 2,
    'ytick.major.width': 0.5,
    'ytick.major.pad': 1,
    'ytick.minor.size': 1,
    'ytick.minor.width': 0.5,
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.labelpad': 1,
    'axes.titlesize': 7,
    'axes.titlepad': 2,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5
}
plt.rcParams.update(style_dict)


def add_arrow(line, position=None, direction='right', size=15,
              color=None, arrowstyle='->', num=1):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   (x,y)-position of the arrow. If None, min*1.007 of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    if position is None:
        position = np.min(xdata)
    # find closest index
    # start_ind = np.argmin(np.absolute(xdata - position))  # - 1500
    start_ind = np.argmin(np.linalg.norm(np.stack((xdata, ydata)) -
                                         np.array(position).reshape(2, 1),
                                         axis=0))
    print('Verify that this has changed since, offset?')
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    for ii in range(num):
        strt_ = start_ind + int(ii*750)
        end_ = end_ind + int(ii*750)
        line.axes.annotate('',
                           xytext=(xdata[strt_], ydata[strt_]),
                           xy=(xdata[end_], ydata[end_]),
                           arrowprops=dict(arrowstyle=arrowstyle, color=color),
                           size=size)


def cm_to_inches(vals):
    return [0.393701*ii for ii in vals]


def add_sizebar(ax, size, loc=8, size_vertical=2, text=None):
    if not text:
        text = str(size) + ' ms'
    asb = AnchoredSizeBar(ax.transData,
                          int(size),
                          text,
                          loc=loc,
                          pad=0.5, borderpad=.4, sep=5,
                          frameon=False,
                          size_vertical=size_vertical)
    ax.add_artist(asb)
    return ax


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)


def add_logticks(axx, xax=True, ticks=[1, 10, 100, 1000]):
    if xax:
        axx.set_xticks(ticks)
        x_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1,
                             numticks=10)
        axx.xaxis.set_minor_locator(x_minor)
        axx.xaxis.set_minor_formatter(NullFormatter())
    else:
        axx.set_yticks(ticks)
        y_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1,
                             numticks=10)
        axx.yaxis.set_minor_locator(y_minor)
        axx.yaxis.set_minor_formatter(NullFormatter())
    return axx


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


snc = {'ros': rotenone_inf, 'atp_bl': 25, 'Q': 30}
dfb = {'ros_sog': minisog_inf, 'atp_bl': [30], 'Q': 20, 'ros_aox': aox_inf}

def_colors = {'nad': Set2_8.hex_colors[6],
              'pyr': Set2_8.hex_colors[7],
              'atp': Set2_8.hex_colors[0],
              'psi': Set2_8.hex_colors[1],
              'ln1': Set1_8.hex_colors[1],
              'ln2': Set1_8.hex_colors[3],
              'ln3': Set1_8.hex_colors[7],
              'ln4': '#1f77b4',
              'ln5': '#ff7f0e',
              'ln6': '#e377c2',
              'ln7': Set3_12.hex_colors[4],
              'ln8': Set3_12.hex_colors[5],
              'ln9': Set3_12.hex_colors[6],
              'ln10': BuPu_5.hex_colors[1],
              'ln11': BuPu_5.hex_colors[2],
              'ln12': BuPu_5.hex_colors[3],
              'ln13': RdPu_5.hex_colors[1],
              'ln14': RdPu_5.hex_colors[2],
              'ln15': RdPu_5.hex_colors[3],
              'ln16': Burg_5.hex_colors[0],
              'ln17': Burg_5.hex_colors[2],
              'ln18': Burg_5.hex_colors[4],
              'ln19': YlGn_5.hex_colors[1],
              'ln20': YlGn_5.hex_colors[2],
              'ln21': YlGn_5.hex_colors[3],
              'ros': '#1f77b4',
              'hyp': '#ca0020',
              'teri': '#008000',
              'aox': '#f04e4d',
              'ret': '#4d9221',
              'fet': '#c51b7d',
              'SD': '#62a73b',
              'minisog': '#3589d6',
              'park1': '#ff0000',
              'park2': '#0000ff'}

colormap_dict = {'Continuous': '#f4cae4',  # Pastel2_8 cb2
                 'Bursting': '#f1e2cc',
                 'Regular': '#cccccc',
                 'Silent': '#fdcdac',
                 'Adapting': '#e6f5c9'}

ln_cols_mcu = [def_colors['ln1'], def_colors['ln2'], def_colors['ln3']]
ln_cols_ros2 = [def_colors['ln7'], def_colors['ln8'], def_colors['ln9']]
ln_cols_ros = [def_colors['ln16'], def_colors['ln17'], def_colors['ln18']]
# ln_cols_ros2 = [def_colors['ln19'], def_colors['ln20'], def_colors['ln21']]
ln_cols_tau = [def_colors['ln13'], def_colors['ln14'], def_colors['ln15']]
ln_cols_q = [def_colors['ln10'], def_colors['ln11'], def_colors['ln12']]
ln_cols_fet = ['#a6761d', '#e6ab02']
ln_cols_ret = ['#1f78b4', '#a6cee3', 'k']
ln_cols_rise = [def_colors['ln1'], def_colors['ln2'], def_colors['ln3']]
