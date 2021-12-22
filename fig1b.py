import numpy as np

from utils import Recorder, Q_nak
from mitochondria import Mito
from steady_state import get_steady_state
from gates import ros_inf, get_ros

import figure_properties as fp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def spike_quanta(baseline_atp, q):
    '''Perturbation due to a spike'''
    dt = 0.01
    time = 500
    factor = 0.1e-4
    tt = np.arange(0, time, dt)
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # state 4 - wait till we reach here
    Q_val = Q_nak(tt, q)
    spike_val = np.zeros_like(tt)
    t_start = 150
    spike_val[int(t_start/dt):] += Q_val[:len(spike_val[int(t_start/dt):])]
    rec_vars_list = ['atp', 'psi', 'k_ant']
    m_record = Recorder(m, rec_vars_list, time, dt)
    for ii, tt in enumerate(np.arange(0, time, dt)):
        try:
            m.update_vals(dt, atp_cost=spike_val[ii],
                          leak_cost=spike_val[ii]*factor)
        except IndexError:
            m.update_vals(dt, leak_cost=0, atp_cost=0)
        m_record.update(ii)
    return m_record


def metabolic_spikes(ax):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    ros_vals = np.zeros_like(bls)
    for ii, bl in enumerate(bls):
        ros_vals[ii] = ros_inf(atp(bl), psi(bl))
    ax.semilogx(bls, ros_vals, label='0 Hz', lw=1, c='k')
    ln_cols = [fp.def_colors['ln1'],
               fp.def_colors['ln2'],
               fp.def_colors['ln3']]
    for jj, freq in enumerate([5, 10, 20]):
        b_test, vals = run_sim(freq, spike_quanta=30)
        ax.semilogx(b_test, vals, label=str(int(freq))+' Hz', marker='o',
                    c=ln_cols[jj],
                    lw=0.5, markersize=2)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xlabel('Non-spiking costs (%s)' % kANT_units)
    ax.set_ylabel(r'ROS level (a.u.)')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend(frameon=False, handlelength=1,
              bbox_to_anchor=(0.2, .7, .6, .2),
              loc='lower center', ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def run_sim(test_freq, spike_quanta, psi_fac=0.1e-4, ros=get_ros(), tau_Q=100):
    fname_list = [test_freq, spike_quanta, psi_fac, ros.name, tau_Q]
    filename = '_'.join([str(yy) for yy in fname_list])
    filename += '.npz'
    try:
        kk = np.load('./spike_compensation/'+filename)
        bls, ros_metb_spikes = kk['bls'], kk['ros_metb_spikes']
    except FileNotFoundError:
        print('No prev compute found, running sim now')
        bls = np.geomspace(1, 1000, 20)
        ros_metb_spikes = np.zeros_like(bls)
        for ij, mito_baseline in enumerate(bls):
            print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
            print('Psi factor: ', psi_fac)
            dt = 0.01
            time = 5000
            t = np.arange(0, time, dt)
            qdur = 1000
            qtime = np.arange(0, qdur, dt)
            this_q = Q_nak(qtime, fact=spike_quanta, tau_Q=tau_Q)
            qlen = len(this_q)
            mi = Mito(baseline_atp=mito_baseline)
            mi.steadystate_vals(time=1000)
            ros.init_val(mi.atp, mi.psi)
            spike_expns = np.zeros_like(t)
            test_isi = 1000 / test_freq
            test_isi_indx = int(test_isi / dt)
            num_spikes = int(time / test_isi)
            for sp in range(1, num_spikes+1):
                sp_idx = test_isi_indx*sp
                try:
                    spike_expns[sp_idx:sp_idx+qlen] += this_q
                except ValueError:
                    spike_expns[sp_idx:] += this_q[:len(spike_expns[sp_idx:])]
            ros_vals = np.zeros_like(t)
            for i in range(len(t)):
                mi.update_vals(dt,
                               atp_cost=spike_expns[i],
                               leak_cost=spike_expns[i]*psi_fac)
                ros.update_vals(dt, mi.atp, mi.psi,
                                spike_expns[i]+mito_baseline)
                ros_vals[i] = ros.get_val()
            ros_metb_spikes[ij] = np.mean(ros_vals)
        np.savez('./spike_compensation/'+filename,
                 bls=bls, ros_metb_spikes=ros_metb_spikes)
    return bls, ros_metb_spikes


def figure_steady_state_simpler(ax1):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    ax1.plot(bls, atp(bls), label=r'$ATP_M$', lw=1.5,
             color=fp.def_colors['atp'])
    ax1.plot(bls, psi(bls), label=r'$\Delta\psi$', lw=1.5,
             color=fp.def_colors['psi'])

    ax1.plot(bls, nad(bls), label='NAD+', lw=1.5,
             color=fp.def_colors['nad'])
    ax1.plot(bls, pyr(bls), label=r'Pyruvate', lw=1.5,
             color=fp.def_colors['pyr'])
    ax1.set_ylabel('(a.u.)')
    # ax1.set_title('Substrate conc. at steady state')
    ax1.set_ylim(0, 2)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$ATP_C \rightarrow ADP_C$ (%s)' % kANT_units+'\n\n')
    ax1.legend(frameon=False, handlelength=0.7,
               bbox_to_anchor=(0., .7, .5, .2),
               loc='lower left',
               ncol=2)
    # ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.55, 0.01))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_yticks([0, 0.5, 1, 1.5, 2])
    ax1.plot(30, 0, marker='*', clip_on=False, color='k',
             markersize=7.5, markeredgecolor='none')
    ax1.plot(150, 0, marker='*', clip_on=False, color='gold', markersize=7,
             markeredgecolor='k', markeredgewidth=0.5,  zorder=10)
    return ax1


kANT_units = '10$^{-3}$/s'
#  half a column size is
figsize = fp.cm_to_inches([8.9, 5])
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(1, 2, figure=fig,
                       wspace=0.5,
                       width_ratios=[1, 1])

ax_steadys = fig.add_subplot(gs[0, 0])
ax_steadys = figure_steady_state_simpler(ax_steadys)
ax_steadys = fp.add_logticks(ax_steadys)

ax_compensate = fig.add_subplot(gs[0, 1])
ax_compensate = metabolic_spikes(ax_compensate)
ax_compensate.spines['top'].set_visible(False)
ax_compensate.spines['right'].set_visible(False)
ax_compensate = fp.add_logticks(ax_compensate)

fp.align_axis_labels([ax_steadys, ax_compensate],
                     axis='x', value=-0.14)

gs.tight_layout(fig)
plt.savefig('Figure1b.png', dpi=300)
# plt.show()

