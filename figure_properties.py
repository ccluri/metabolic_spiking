import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set2_8
from gates import hypoxia_inf, minisog_inf, aox_inf, teri_inf, ageing_inf
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from enum import Enum

# plt.style.use('seaborn-talk')

# plt.rcParams.update({
#     'xtick.labelsize': 10,
#     'xtick.major.size': 10,
#     'ytick.labelsize': 10,
#     'ytick.major.size': 10,
#     'font.size': 10,
#     'axes.labelsize': 12,
#     'axes.titlesize': 12,
#     'axes.titlepad' : 5,
#     'legend.fontsize': 12,
#     # 'figure.subplot.wspace': 0.4,
#     # 'figure.subplot.hspace': 0.4,
#     # 'figure.subplot.left': 0.1,
# })

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'xtick.labelsize': 5,
    'xtick.major.size': 3,
    'xtick.major.width': 0.5,
    'xtick.major.pad': 1,
    'xtick.minor.size': 1,
    'xtick.minor.width': 0.5,
    'ytick.labelsize': 5,
    'ytick.major.size': 3,
    'ytick.major.width': 0.5,
    'ytick.major.pad': 1,
    'ytick.minor.size': 1,
    'ytick.minor.width': 0.5,
    'font.size': 5,
    'axes.labelsize': 5,
    'axes.labelpad': 1,
    'axes.titlesize': 5,
    'axes.titlepad': 5,
    'legend.fontsize': 5,
    'axes.linewidth': 0.5
    
    # 'figure.subplot.wspace': 0.4,
    # 'figure.subplot.hspace': 0.4,
    # 'figure.subplot.left': 0.1,
})


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


        
test_bl_atps = [10, 50, 100, 500]
test_bl_clrs = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
test_bl_markers = ['o', '^', 's', 'p'] # no of sides = atp consumption level

hypoxia = {'ros':hypoxia_inf ,'atp_bl':30, 'Q':25}
dfb = {'ros_sog':minisog_inf, 'atp_bl':[30], 'Q':20, 'ros_aox':aox_inf}

# ca1 = {'ros':teri_inf, 'atp_bl':[10.7779418 ,  29.30700118,  27.45005909,  33.83886601,
#                                  9.74197384,  23.27841053,  21.64849983,  24.28203825,
#                                  29.0542684 ,  15.54064254], # np.random.randn(10)*10 + 20
#        'Q':[14, 23, 20, 10, 16, 11, 27, 21, 26, 11]} # np.random.randint(10, 30, 10)

r = np.random.RandomState(10)
ca1 = {'ros':teri_inf, 'atp_bl': r.randn(100)*10 + 30, # np.random.randn(10)*10 + 20
       'Q':r.randint(20, 30, 100)} # np.random.randint(10, 30, 10)
snc = {'ros':ageing_inf, 'atp_bl':25, 'Q':30}


def_colors = {'nad': Set2_8.hex_colors[6],
              'pyr': Set2_8.hex_colors[7],
              'cit': Set2_8.hex_colors[5],
              'atp': Set2_8.hex_colors[0],
              'psi': Set2_8.hex_colors[1],
              'ros': '#1f77b4',
              'hyp': '#ca0020',
              'teri': '#008000',
              'aox': '#f04e4d',
              'SD': '#62a73b',
              'minisog': '#3589d6',
              'park1': '#ff0000',
              'park2': '#0000ff'}


class TestCases(object):
    def __init__(self, bl, q, start, end, sizebar=100,
                 arrowsx=[], texts={}, showyticks=False):
        self.bl = bl
        self.q = q
        self.start = start
        self.end = end
        self.sizebar = sizebar  # ms
        self.arrowsx = arrowsx
        self.texts = texts
        self.showyticks = showyticks

# # ros_tau1
t_regular = TestCases(38, 35, 1150, 1550, arrowsx=[1212], showyticks=True)
t_fast_regular = TestCases(34, 35, 1050, 1450, arrowsx=[1114, 1380],
                           texts={'1': 1114, '2': 1380})
t_bursting = TestCases(34, 5, 1200, 1600, texts={'1': 1280, '2': 1285,
                                                 '3': 1290, '10': 1330})
t_continuous = TestCases(34, 0.15, 1500, 1600, sizebar=10)

# t_regular = TestCases(40, 25, 1150, 1550, arrowsx=[1212], showyticks=True)
# t_fast_regular = TestCases(30, 25, 1050, 1450, arrowsx=[1114, 1380],
#                            texts={'1': 1114, '2': 1380})
# t_bursting = TestCases(30, 5, 1200, 1600, texts={'1': 1280, '2': 1285,
#                                                  '3': 1290, '10': 1330})
# t_continuous = TestCases(30, 1, 1500, 1600, sizebar=10)

# t_regular.inset_vals(0.66, 0.28, 0.24, 0.32) # 38, 35

# firing_tests = [t_regular, t_fast_regular]
firing_tests = [t_regular, t_fast_regular,
                t_bursting, t_continuous]
