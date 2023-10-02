import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from IPython.display import display, clear_output

from gates import ros_inf, get_ros
from mitochondria import Mito
import figure_properties as fp
from utils import Q_nak, Recorder
from steady_state import get_steady_state
from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))


kANT_units = '10$^{-3}$/s'
atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
bls = np.geomspace(1, 1000, 100)
ros_vals = np.zeros_like(bls)
for ii, bl in enumerate(bls):
    ros_vals[ii] = ros_inf(atp(bl), psi(bl))

ss_output = widgets.Output()


def ros_ss(min_atp=10, max_atp=200):
    with ss_output:
        clear_output()
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = plt.subplot(111)
        min_rect = Rectangle(xy=(1, 0), height=1, width=min_atp,
                             facecolor='gray', edgecolor='w', fill=True, alpha=0.3)
        max_rect = Rectangle(xy=(max_atp, 0), height=1, width=bls[-1]-max_atp,
                             facecolor='gray', edgecolor='w', fill=True, alpha=0.3)
        ax.add_patch(min_rect)
        ax.add_patch(max_rect)
        plt.semilogx(bls, ros_vals, lw=2, c='k', ls=':',
                     label='Theoretical exposure')
        strt = np.argwhere(bls > min_atp)[0][0]
        end = np.argwhere(bls > max_atp)[0][0]
        plt.semilogx(bls[strt:end], ros_vals[strt:end], lw=2, c='k',
                     label='Physiological experience')
        ax.set_ylim(-0.01, 1.)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel(r'Non-spiking costs (%s)' % kANT_units)
        ax.set_ylabel(r'ROS level (a.u.)')
        plt.legend(frameon=False, loc='upper right')
        plt.show()
    return


style = {'description_width': 'initial'}
min_kant = widgets.IntSlider(value=10, min=1, max=50, step=1,
                             description='Lifetime min Non-spiking costs:',
                             disabled=False,
                             continuous_update=False,
                             orientation='horizontal',
                             readout=True,
                             readout_format='d', style=style,
                             layout=widgets.Layout(width='400px'))

max_kant = widgets.IntSlider(value=200,
                             min=50,
                             max=1000,
                             step=1,
                             description='Lifetime max Non-spiking costs :',
                             disabled=False,
                             continuous_update=False,
                             orientation='horizontal',
                             readout=True,
                             readout_format='d', style=style,
                             layout=widgets.Layout(width='400px'))
##########################################################################


curr_cost1 = widgets.IntSlider(value=30,
                               min=min_kant.value,
                               max=max_kant.value,
                               step=1,
                               description='Current Non-spiking costs :',
                               disabled=False,
                               continuous_update=False,
                               orientation='horizontal',
                               readout=True,
                               readout_format='d', style=style,
                               layout=widgets.Layout(width='500px'))


def update_min_range(*args):
    curr_cost1.min = min_kant.value


def update_max_range(*args):
    curr_cost1.max = max_kant.value


curr_cost1.observe(update_min_range, 'value')
curr_cost1.observe(update_max_range, 'value')

rosss_output = widgets.Output()

def ros_ss_steady(bl):
    with rosss_output:
        clear_output
        fig = plt.figure(figsize=(10, 5), dpi=100)
        ax = plt.subplot(121)
        min_atp = min_kant.value
        max_atp = max_kant.value
        min_rect = Rectangle(xy=(1, 0), height=1, width=min_atp,
                             facecolor='gray', edgecolor='w',
                             fill=True, alpha=0.3)
        max_rect = Rectangle(xy=(max_atp, 0), height=1, width=bls[-1]-max_atp,
                             facecolor='gray', edgecolor='w', fill=True, alpha=0.3)
        ax.add_patch(min_rect)
        ax.add_patch(max_rect)
        plt.semilogx(bls, ros_vals, lw=2, c='k', ls=':')
        strt = np.argwhere(bls > min_atp)[0][0]
        end = np.argwhere(bls > max_atp)[0][0]
        ax.semilogx(bls[strt:end], ros_vals[strt:end], lw=2, c='k')
        ax.set_ylim(-0.01, 1.)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel(r'Non-spiking costs (%s)' % kANT_units)
        ax.set_ylabel(r'ROS level (a.u.)')
        ax.plot(bl, -0.01, marker='*', clip_on=False, color='k', markersize=20,
                markeredgecolor='none')
        ax2 = plt.subplot(122)
        ax2.bar([0, 1, 2, 3, 4],
                [atp(bl), psi(bl), nad(bl), pyr(bl),
                 ros_inf(atp(bl), psi(bl))],
                color=[fp.def_colors['atp'], fp.def_colors['psi'],
                       fp.def_colors['nad'], fp.def_colors['pyr'], 'r'],
                width=0.5)
        ax2.set_ylim(0, 1.5)
        ax2.set_xticks([0, 1, 2, 3, 4])
        ax2.set_yticks([0, 0.5, 1, 1.5])
        ax2.set_xticklabels(['ATP', 'PSI', 'NAD+', 'Pyr', 'ROS'])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.show()
    return
#######################################################


fact = widgets.FloatSlider(value=0.1,
                           min=0.01,
                           max=0.5,
                           step=0.01,
                           description='Per-spike costs:',
                           disabled=False,
                           continuous_update=False,
                           orientation='horizontal',
                           readout=True,
                           readout_format='.2f', style=style,
                           layout=widgets.Layout(width='500px'))

tau_Q = widgets.FloatSlider(value=100,
                            min=10,
                            max=300,
                            step=10,
                            description='Q decay (ms):',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            readout_format='d', style=style,
                            layout=widgets.Layout(width='500px'))

tau_rise = widgets.FloatSlider(value=5,
                               min=0.1,
                               max=10,
                               step=0.1,
                               description='Q rise (ms):',
                               disabled=False,
                               continuous_update=False,
                               orientation='horizontal',
                               readout=True,
                               readout_format='.1f', style=style,
                               layout=widgets.Layout(width='500px'))


def reset_button1(defaults={}):
    def on_button_clicked(_):
        for k, v in defaults.items():
            k.value = v
    button = widgets.Button(description='Default values')
    button.on_click(on_button_clicked)
    display(button)


sshape = widgets.Output()
    
def spike_shape(bl=45, q=0.1, tau_Q=100, tau_rise=5):
    dt = 0.01
    time = 1000
    t_start = 150
    tt = np.arange(0, time, dt)
    Q_val = Q_nak(tt, fact=q, tau_rise=tau_rise, tau_Q=tau_Q)
    spike_val = np.zeros_like(tt) + atp(bl)
    spike_val[int(t_start/dt):] -= Q_val[:len(spike_val[int(t_start/dt):])]
    with sshape:
        clear_output
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = plt.subplot(111)
        ax.plot(tt, spike_val, lw=2, c='k')
        ax.plot(-100, atp(bl), marker='*', clip_on=False, color='k', markersize=20,
                markeredgecolor='none')
        ax.set_xlim(-100, 1000)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('ATP cytosol (a.u.)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        plt.show()
#######################################################

ca_fact = widgets.FloatSlider(value=1e-3,
                              min=0,
                              max=1e-2,
                              step=1e-4,
                              description='Ca per (psi factor) spike:',
                              disabled=False,
                              continuous_update=False,
                              orientation='horizontal',
                              readout=True,
                              readout_format='.4f', style=style,
                              layout=widgets.Layout(width='500px'))


def reset_button2(defaults={}):
    def on_button_clicked(_):
        for k, v in defaults.items():
            k.value = v
    button2 = widgets.Button(description='Default values')
    button2.on_click(on_button_clicked)
    display(button2)


def ros_land_dummy():
    ATP = np.arange(0, 1.05, 0.05)
    PSI = np.arange(0, 1.05, 0.05)
    ATP, PSI = np.meshgrid(ATP, PSI)
    ROS = (ATP*PSI) + ((1-ATP)*(1-PSI))
    return ATP, PSI, ROS**3


ATP, PSI, ROS = ros_land_dummy()
ros = get_ros()
rec_vars_list = ['atp', 'psi', 'k_ant', 'nad', 'pyr']

squanta = widgets.Output()

def spike_quanta(bl, q=0.1, tau_Q=tau_Q, tau_rise=tau_rise, f_mcu=1e-3):
    dt = 0.01
    time = 1000
    t_start = 150
    tt = np.arange(0, time, dt)
    m = Mito(baseline_atp=bl)
    m.steadystate_vals(time=2000)  # state 4 - wait till we reach here
    ros.init_val(m.atp, m.psi)
    spike_val = np.zeros_like(tt) + m.atp
    leak_val = np.zeros_like(tt)
    ros_vals = np.zeros_like(tt)
    Q_val = Q_nak(tt, 1, tau_Q=tau_Q, tau_rise=tau_rise)
    spike_val[int(t_start/dt):] -= Q_val[:len(spike_val[int(t_start/dt):])]*q                                                                                                
    leak_val[int(t_start/dt):] += Q_val[:len(leak_val[int(t_start/dt):])]*f_mcu*q                                                                                            
    m_record = Recorder(m, rec_vars_list, time, dt)
    for ii, ti in enumerate(tt):
        try:
            m.update_vals(dt, atp_cost=spike_val[ii],
                          leak_cost=leak_val[ii])
        except IndexError:
            m.update_vals(dt, leak_cost=0, atp_cost=m.atp)
        m_record.update(ii)
        ros.update_vals(dt, m.atp, m.psi,
                        m.k_ant*1000)   # keeping with the prev. ver.
        ros_vals[ii] = ros.get_val()
    with squanta:
        clear_output
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100,
                                     gridspec_kw={'width_ratios': [1, 2]})
        ax1.contourf(ATP, PSI, ROS, 50, cmap=cm.Reds)
        ax1.plot(m_record.out['atp'], m_record.out['psi'],
                 lw=1.5, c='k', alpha=1)
        ax1.plot(m_record.out['atp'][0], m_record.out['psi'][0],
                 marker='*', c='k', clip_on=False, markersize=20,
                 markeredgecolor='none')
        ax1.set_xlabel('ATP Mito (a.u)')
        ax1.set_ylabel('PSI (a.u)')
        # ax1.set_aspect('equal')

        ax2.plot(tt, m_record.out['atp'], c=fp.def_colors['atp'],
                 lw=2, label='ATP')
        ax2.plot(tt, m_record.out['psi'], c=fp.def_colors['psi'],
                 lw=2, label='PSI')
        ax2.plot(tt, m_record.out['nad'], c=fp.def_colors['nad'],
                 lw=2, label='NAD+')
        ax2.plot(tt, m_record.out['pyr'], c=fp.def_colors['pyr'],
                 lw=2, label='Pyr')
        ax2.plot(tt, ros_vals, c='r', lw=2, label='ROS')
        plt.legend(frameon=False, ncol=5, loc='upper center')
        ax2.set_ylim(0, 1.5)
        ax2.set_yticks([0, 0.5, 1, 1.5])
        ax2.set_ylabel('Substrate conc. (a.u)')
        ax2.set_xlabel('Time (ms)')
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.show()
    return

###############################################
