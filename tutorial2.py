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
ros_vals = []
atp_vals = []
for ii, bl in enumerate(bls):
    ros_vals.append(ros_inf(atp(bl), psi(bl)))
    atp_vals.append(atp(bl))
ros_vals = np.array(ros_vals)
atp_vals = np.array(atp_vals)
K_idx = np.where(ros_vals == np.min(ros_vals))
atp_at_min = atp_vals[K_idx]
ms = ros_vals*(atp_vals-atp_at_min)
    

def ros_ss(min_atp=10, max_atp=200, bl=30, theta_ret=0.025, theta_fet=-0.05):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = plt.subplot(111)
    min_rect = Rectangle(xy=(1, -0.5), height=1, width=min_atp,
                         facecolor='gray', edgecolor='w', fill=True, alpha=0.3)
    max_rect = Rectangle(xy=(max_atp, -0.5), height=1, width=bls[-1]-max_atp,
                         facecolor='gray', edgecolor='w', fill=True, alpha=0.3)
    ax.add_patch(min_rect)
    ax.add_patch(max_rect)
    ax.semilogx(bls, ms, lw=0.5,
                color='k', label='MS=ROS x $\partial$ATP')
    ax.plot(bl, -0.3, marker='*', clip_on=False, color='k', markersize=20,
            markeredgecolor='none')
    ms_v = ros_inf(atp(bl), psi(bl)) * (atp(bl)-atp_at_min)
    ax.plot([bl, bl], [-0.3, ms_v[0]], c='gray', ls='--', lw=1)
    ax.plot([1, bl], [ms_v[0], ms_v[0]], ls='--', c='gray', lw=1)
    if ms_v > theta_ret:
        s = 'RET Exposure'
        c = fp.def_colors['ret']
    elif ms_v < theta_fet:
        s = 'FET Exposure'
        c = fp.def_colors['fet']
    else:
        s = 'Safe Zone'
        c = 'gray'
    ax.text(2, ms_v[0]+0.02, s=s, c=c, fontsize=10,
            transform=ax.transData, va='center')
    strt = np.argwhere(bls > min_atp)[0][0]
    end = np.argwhere(bls > max_atp)[0][0]
    ax.plot([min_atp, max_atp], [theta_ret, theta_ret], lw=1,
            ls='-', color=fp.def_colors['ret'], zorder=-1)
    ax.plot([min_atp, max_atp], [theta_fet, theta_fet], lw=1,
            ls='-', color=fp.def_colors['fet'], zorder=-1)
    ax.text(65, theta_ret+0.05, s=r'$\theta_{RET}$', ha='right',
            color=fp.def_colors['ret'], va='center',
            transform=ax.transData, fontsize=7, clip_on=False)
    ax.text(65, theta_fet-0.05, s=r'$\theta_{FET}$', ha='right',
            color=fp.def_colors['fet'], va='center',
            transform=ax.transData, fontsize=7, clip_on=False)
    # ax.text(65, (theta_ret + theta_fet)/2, s=r'Safe zone', ha='right',
    #         color='gray', va='center', alpha=0.4,
    #         transform=ax.transData, fontsize=10, clip_on=False)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([-0.3, 0, 0.3])
    ax.set_xlabel(r'Non-spiking costs (%s)' % kANT_units)
    ax.set_ylabel(r'Metabolic Signal (a.u.)')
    plt.legend(frameon=False, loc='upper right')
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

ret_adj = widgets.FloatSlider(value=0.025,
                              min=0.01,
                              max=0.05,
                              step=0.005,
                              description='RET protection Max:',
                              disabled=False,
                              continuous_update=False,
                              orientation='horizontal',
                              readout=True,
                              readout_format='.3f', style=style,
                              layout=widgets.Layout(width='400px'))

fet_adj = widgets.FloatSlider(value=-0.05,
                              min=-0.08,
                              max=0.0,
                              step=0.005,
                              description='FET protection Min:',
                              disabled=False,
                              continuous_update=False,
                              orientation='horizontal',
                              readout=True,
                              readout_format='.3f', style=style,
                              layout=widgets.Layout(width='400px'))

    
def reset_button1(defaults={}):
    def on_button_clicked(b):
        for k, v in defaults.items():
            k.value = v
    button = widgets.Button(description='Default values')
    button.on_click(on_button_clicked)
    display(button)

#############################################################


evaluate_output = widgets.Label('Unknown')
evaluate_button = widgets.Button(description='Evaluate status')
hbox1 = widgets.HBox([evaluate_button, evaluate_output])

def eval_curr_status(b):
    with retprot_output:
        clear_output()
    bl = curr_cost1.value
    curr_ros = ros_inf(atp(bl), psi(bl))
    curr_atp = atp(bl)
    ms = curr_ros*(curr_atp-atp_at_min)
    if ms > ret_adj.value:
        evaluate_output.value = 'RET Exposure'
        retprot_button.disabled = False
        fetprot_button.disabled = True
    elif ms < fet_adj.value:
        evaluate_output.value = 'FET Exposure'
        retprot_button.disabled = True
        fetprot_button.disabled = False
    else:
        evaluate_output.value = 'Safe zone'
        retprot_button.disabled = True
        fetprot_button.disabled = True
    return

evaluate_button.on_click(eval_curr_status)

retprot_button = widgets.Button(description='RET Protection', disabled=True)
retprot_output = widgets.Output()
fetprot_button = widgets.Button(description='FET Protection', disabled=True)

#########################################################


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
                               

fact = widgets.FloatSlider(value=0.1,
                           min=0.01,
                           max=0.2,
                           step=0.01,
                           description='Per-spike costs:',
                           disabled=False,
                           continuous_update=False,
                           orientation='horizontal',
                           readout=True,
                           readout_format='.2f', style=style,
                           layout=widgets.Layout(width='500px'))
                        
                              
def reset_button2(defaults={}):
    def on_button_clicked(_):
        for k, v in defaults.items():
            k.value = v
    button2 = widgets.Button(description='Default values')
    button2.on_click(on_button_clicked)
    display(button2)
    
    
def spike_shape(bl=45, q=0.1, tau_rise=5):
    dt = 0.01
    time = 1000
    t_start = 150
    tt = np.arange(0, time, dt)
    Q_val = Q_nak(tt, fact=q, tau_rise=tau_rise, tau_Q=100)
    spike_val = np.zeros_like(tt) + atp(bl)
    spike_val[int(t_start/dt):] -= Q_val[:len(spike_val[int(t_start/dt):])]
    fig = plt.figure(figsize=(3, 3), dpi=100)
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
    
    
#######################################################
refrac_sel = widgets.IntSlider(value=5, min=1, max=15, step=1,
                               description='Refractory period (ms):',
                               disabled=False,
                               continuous_update=False,
                               orientation='horizontal',
                               readout=True,
                               readout_format='d', style=style,
                               layout=widgets.Layout(width='400px'))


def neat_axes(axs):
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        
def clear_ticks(axs, x=True, y=False):
    for ax in axs:
        if x:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if y:
            ax.set_yticks([])
            ax.set_yticklabels([])

            
def fetch_rel_spikes(t, spks, arr):
    arr_val_at_spike = []
    for sp in spks:
        arr_val_at_spike.append(arr[np.where(t == sp)[0][0]])
    return arr_val_at_spike


def add_spikes(ax, t, spikes, arr):
    y_coords = fetch_rel_spikes(t, spikes, arr)
    ax.plot(spikes, y_coords, marker='o',
            c='k', lw=0, zorder=-ii,
            markersize=5,  markerfacecolor='k',
            markeredgecolor='r', markeredgewidth=0.2)

        
def ret_run_sim(b):
    mito_baseline = curr_cost1.value
    spike_quanta = fact.value
    psi_fac = 1e-3
    refrac = refrac_sel.value
    dt = 0.01
    time = 1250
    t = np.arange(0, time, dt)
    qdur = 1000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, fact=1, tau_rise=tau_rise.value)
    qlen = len(this_q)
    ros = get_ros()
    # Mitochondria
    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=2000)
    ros.init_val(mi.atp, mi.psi)
    r_mito = Recorder(mi, ['atp'], time, dt)
    spike_expns = np.zeros_like(t) + mi.atp
    leak_expns = np.zeros_like(t)
    ross_vals = np.zeros_like(t)
    ms_vals = np.zeros_like(t)
    spikes = []  # fake spikes
    fire_mode = False
    spiked = False
    elapsed = 0
    for i in range(len(t)):
        mi.update_vals(dt,
                       atp_cost=spike_expns[i],
                       leak_cost=leak_expns[i])
        ros.update_vals(dt, mi.atp, mi.psi,
                        mi.k_ant*1000)
        ross_vals[i] = ros.get_val()
        msig = ros.get_val()*(mi.atp - atp_at_min)
        ms_vals[i] = msig
        if msig > ret_adj.value:  # RET ROS
            fire_mode = True
        else:
            fire_mode = False
        if fire_mode and not spiked:
            spiked = True
            elapsed = 0
            spikes.append(t[i])
            try:
                spike_expns[i:i+qlen] -= this_q*spike_quanta
                leak_expns[i:i+qlen] += this_q*psi_fac*spike_quanta
            except ValueError:
                spike_expns[i:] -= this_q[:len(spike_expns[i:])]*spike_quanta
                leak_expns[i:] += this_q[:len(leak_expns[i:])]*psi_fac*spike_quanta
        else:
            if elapsed < refrac:
                elapsed += dt
            else:
                spiked = False
        r_mito.update(i)
    
    with retprot_output:
        clear_output()
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax0 = plt.subplot(411)
        for sp in spikes:
            ax0.plot([sp, sp], [0.1, 0.2], lw=0.5, c='k')
        ax0.set_ylim([0, 0.3])
        ax0.set_ylabel('Spikes')
        ax0.set_yticklabels([])
        ax0.set_ylabel([])
        ax0.set_xlim([100, 1200])
        ax = plt.subplot(412)
        ax.plot(t, ross_vals, c='r', lw=1)
        add_spikes(ax, t, spikes, ross_vals)
        ax.set_ylabel('ROS (au)')
        ax.set_ylim([0, 0.25])
        ax.set_xlim([100, 1200])
        ax2 = plt.subplot(413)
        ax2.plot(t, r_mito.out['atp'], c=fp.def_colors['atp'], lw=1)
        add_spikes(ax2, t, spikes, r_mito.out['atp'])
        ax2.set_ylabel('ATP (au)')
        ax2.set_ylim([.7, 1])
        ax2.set_xlim([100, 1200])
        ax3 = plt.subplot(414)
        ax3.plot(t, ms_vals, c='k', lw=1)
        ax3.plot(t, [ret_adj.value]*len(t),
                 c=fp.def_colors['ret'],
                 ls='--', lw=0.5)
        add_spikes(ax3, t, spikes, ms_vals)
        ax3.text(500, ret_adj.value+0.005, s=r'$\theta_{RET}$', ha='right',
                 color=fp.def_colors['ret'], va='center',
                 transform=ax3.transData, fontsize=7, clip_on=False)
        ax3.set_ylim([-0.01, ret_adj.value+0.01])
        ax3.set_ylabel('MS (au)')
        ax3.set_xlabel('Time (ms)')
        ax3.set_xlim([100, 1200])
        neat_axes([ax0, ax, ax2, ax3])
        clear_ticks([ax0, ax, ax2], x=True)
        plt.show()
    return

retprot_button.on_click(ret_run_sim)
    
