import numpy as np


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

class Recorder():
    def __init__(self, inst, rec_vars_list, time, dt):
        self.inst = inst
        self.time = time
        self.dt = dt
        self.out = {}
        for item in rec_vars_list:
            self.out[item] = np.zeros(int(time/dt))
        self.rec_vars_list = rec_vars_list

    def update(self, ii):
        for item in self.rec_vars_list:
            try:
                self.out[item][ii] = getattr(self.inst, item)
            except AttributeError:  # go deeper
                start = self.inst
                for sub_i in item.split('.'):
                    start = getattr(start, sub_i)
                self.out[item][ii] = start
        return

    def adjust_var(self, item, spike_times, preset):
        tt = np.arange(0, self.time, self.dt)
        idxs = np.where(np.in1d(tt, np.array(spike_times)))[0]
        new_vals = np.array(self.out[item])
        new_vals[idxs] = preset
        self.out[item] = list(new_vals)
        

def fatten_spike_w_current(ax, tt, v, i, col, factor=1):
    from matplotlib.collections import LineCollection
    points = np.array([tt, v]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc1 = LineCollection(segments, colors=[col]*len(segments), linewidths=factor*i)
    ax.add_collection(lc1)
    return ax

def Q_nak(tt, fact=1, tau_Q=100):
    # dt = 0.01
    # tt = np.arange(0, 600, dt)
    vals = 1/(1*np.exp((tt-1)/tau_Q) + 30*np.exp(-(tt-2)/.1))
    # vals = 1/(1*np.exp((tt-1)/200) + 30*np.exp(-(tt-1)/.1))
    nrm = np.max(vals)
    return fact*vals/nrm

def Q_nak2(tt, fact=1):
    # dt = 0.01
    # tt = np.arange(0, 600, dt)
    vals = 1/(1*np.exp((tt-1)/300) + 30*np.exp(-(tt-2)/.1))
    # vals = 1/(1*np.exp((tt-1)/200) + 30*np.exp(-(tt-1)/.1))
    return fact*vals/np.max(vals)


def add_arrow(line, position=None, direction='right', size=15,
              color=None, arrowstyle='->', num=1):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, min*1.007 of xdata is taken
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
    start_ind = np.argmin(np.absolute(xdata - position))  # - 1500
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
    # return xdata[start_ind], ydata[start_ind], xdata[end_ind], ydata[end_ind]


chann_colors = {'Na_P': '#a1d99b',
                'Na_T': '#31a354',
                'K_DR': '#bcbddc',
                'K_A': '#756bb1',
                'K_ATP': '#6baed6',
                'Ca_T': '#fc9272',
                'Ca_L': '#de2d26'}

gate_styles = {'n':'-',
               'm':'-', 'h':'--',
               'p':'-',
               'a':'-', 'b':'--', 'c':'-.', 'f':':',
               'd':'-',
               't':'-', 's':'--', 'q':'-.',
               'l':'-', 'r':'--'}

ros_chann = ['Na_P', 'K_A']
atp_chann = ['K_ATP', 'Ca_T']
# sp_gates = ['p', 'q', 'e', 'f', 'd']


