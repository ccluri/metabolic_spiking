import numpy as np

from utils import Recorder, Q_nak
from mitochondria import Mito
from steady_state import get_steady_state
from lifcell import LIFCell
from gates import ros_inf, get_ros

import figure_properties as fp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from matplotlib import cm


def single_spike(ax0):
    '''Dummy spike'''
    params = {'deltaga': 0, 'refrc': 10,
              'Q': 1, 'init_atp': 1, 'g_nap': 1.25}
    c = LIFCell('test', **params)
    dt = 0.01
    time = 500
    t = np.arange(0, time, dt)
    i_inj = np.zeros_like(t)
    t_start = 150
    t_end = 155
    i_inj[int(t_start/dt):int(t_end/dt)] = 150
    r = Recorder(c, ['v'], time, dt)
    for i in range(len(t)):
        c.i_inj = i_inj[i]
        c.update_vals(dt)
        r.update(i)
    ax0.plot(t, r.out['v'], c='k', lw=0.5)
    ax0.set_ylabel('Memb. Pot.')
    ax0.text(0, 0, '(mV)', va='center', ha='left')
    ax0.set_xlim(-25, 500)
    return ax0


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


def quantum(ax1):
    '''Illustrating baseline plus Q atp->adp'''
    dt = 0.01
    tt = np.arange(0, 500, dt)
    Qval = np.zeros_like(tt)
    vals = Q_nak(tt, 30)
    Qval[int(150/dt):] += vals[:len(Qval[int(150/dt):])]
    lns = ax1.plot(tt, Qval, color='w', alpha=0) # fake line
    points = np.array([tt, Qval]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 500)
    lc = LineCollection(segments, cmap='viridis_r', norm=norm)
    # Set the values used for colormapping
    lc.set_array(tt)
    lc.set_linewidth(1)
    ax1.add_collection(lc)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylim(-10, 40)
    ax1.set_ylabel(r'$ATP_C \rightarrow ADP_C$' +
                   '\n(%s)' % kANT_units)
    ax1.set_yticks([])
    labels = [item.get_text() for item in ax1.get_yticklabels()]
    empty_string_labels = [' ']*len(labels)
    ax1.set_yticklabels(empty_string_labels)
    ax1.plot(-25, 0, marker='*', c='k', clip_on=False, markersize=7.5,
             markeredgewidth=0.5, markeredgecolor='white', zorder=10)
    ax1.text(s='+Q', x=85, y=27.5, fontsize=7)
    fp.add_arrow(lns[0], position=155, color='k', size=6.5)
    fp.add_arrow(lns[0], position=300, color='k', size=6.5)
    ax1.set_xlim(-25, 500)


def ros_land_dummy(ax):
    ATP = np.arange(0, 1.05, 0.05)
    PSI = np.arange(0, 1.05, 0.05)
    ATP, PSI = np.meshgrid(ATP, PSI)
    ROS = (ATP*PSI) + ((1-ATP)*(1-PSI))
    surf = ax.contourf(ATP, PSI, ROS**3, 100, cmap=cm.Reds)
    return surf


def excursion(ax2):
    ros_land_dummy(ax2)
    # plot_bl_curve(ax2)
    baselines = [30, 150]
    star_colors = ['black', 'gold']
    dt = 0.01
    tt = np.arange(0, 500, dt)
    lns = []
    for cc, baseline_atp in zip(star_colors, baselines):
        m_state1 = spike_quanta(baseline_atp=baseline_atp, q=50)
        lns.append(ax2.plot(m_state1.out['atp'], m_state1.out['psi'],
                            lw=1, c='white', alpha=0.2))
        points = np.array([m_state1.out['atp'],
                           m_state1.out['psi']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 500)
        lc = LineCollection(segments, cmap='viridis_r', norm=norm)
        # Set the values used for colormapping
        lc.set_array(tt)
        lc.set_linewidth(1)
        ax2.add_collection(lc)
        if cc == 'gold':
            ax2.plot(m_state1.out['atp'][0], m_state1.out['psi'][0],
                     marker='*', c=cc, clip_on=False, markersize=7,
                     markeredgecolor='k', markeredgewidth=0.5)
        else:
            ax2.plot(m_state1.out['atp'][0], m_state1.out['psi'][0],
                     marker='*', c=cc, clip_on=False, markersize=7.5,
                     markeredgecolor='none')
    fp.add_arrow(lns[0][0], position=0.85, color='k', size=6.5)
    fp.add_arrow(lns[0][0], position=0.74, color='k', size=6.5)
    ax2.set_aspect('equal')
    ax2.set_xlim(0, 1.)
    ax2.set_ylim(0, 1.)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_xticklabels([0, '', 1])
    ax2.set_yticklabels([0, '', 1])
    ax2.set_xlabel('$ATP_M$')
    ax2.set_ylabel(r'$\Delta\psi$', rotation=0)
    return ax2


def ros_ss(ax):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    ros_vals = np.zeros_like(bls)
    for ii, bl in enumerate(bls):
        ros_vals[ii] = ros_inf(atp(bl), psi(bl))
    ax.semilogx(bls, ros_vals, label=r'$ROS_{SS}$', lw=1, c='k')
    ax.plot(30, 0, marker='*', clip_on=False, color='k', markersize=7.5,
            markeredgecolor='none')
    ax.plot(150, 0, marker='*', clip_on=False, color='gold', markersize=7,
            markeredgecolor='k', markeredgewidth=0.5, zorder=10)
    ax.set_ylim(0., 1.)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel(r'Non-spiking costs (%s)' % kANT_units)
    ax.set_ylabel(r'ROS level (a.u.)')
    return ax


def plot_bl_curve(ax):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    ax.plot(atp(bls), psi(bls), ls='-', lw=0.5, c='k')
    ax.plot(atp(150), psi(150), marker='*', markersize=7, c='gold',
            markeredgecolor='k', markeredgewidth=0.5)
    ax.plot(atp(30), psi(30), marker='*', markersize=7.5, c='k',
            markeredgecolor='none')
    return ax


def ros_land(ax, cax=None):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    ATP = np.arange(0, 1.05, 0.05)
    PSI = np.arange(0, 1.05, 0.05)
    ATP, PSI = np.meshgrid(ATP, PSI)
    ROS = (ATP*PSI) + ((1-ATP)*(1-PSI))
    surf = ax.contourf(ATP, PSI, ROS**3, 100, cmap=cm.Reds)
    # past ret color = #009933
    bstyle = mpatches.BoxStyle("Round", pad=0.0,
                               rounding_size=0.)
    retbox = mpatches.FancyBboxPatch((0.55, 0.55), 0.45, 0.45, fill=False,
                                     boxstyle=bstyle,
                                     alpha=0.5, zorder=10, facecolor='None',
                                     edgecolor='#4dac26', linewidth=2)

    fetbox = mpatches.FancyBboxPatch((0.0, 0.0), 0.45, 0.45, fill=False,
                                     boxstyle=bstyle,
                                     alpha=0.5, zorder=10, facecolor='None',
                                     edgecolor='#d01c8b', linewidth=2)
    ax.text(0.07, .5, 'FETROS',
            fontsize=7, color='#d01c8b').set_clip_on(False)
    ax.text(0.65, 1.05, 'RETROS',
            fontsize=7, color='#4dac26').set_clip_on(False)
    retbox.set_clip_on(False)
    fetbox.set_clip_on(False)
    ax.add_patch(retbox)
    ax.add_patch(fetbox)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels([0, '', 1])
    ax.set_yticklabels([0, '', 1])
    ax.set_xlabel(r'$ATP_M$')
    ax.set_ylabel(r'$\Delta\psi$', rotation=0)
    return ax, surf


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
figsize = fp.cm_to_inches([8.9, 13.5])
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.35, 1.35, 1],
                       width_ratios=[1, 1])

ax_rosland = fig.add_subplot(gs[0, 0])
ax_rosland, surf = ros_land(ax_rosland, None)
ax_rosland = plot_bl_curve(ax_rosland)

ax_rosss = fig.add_subplot(gs[0, 1])
ax_rosss = ros_ss(ax_rosss)
ax_rosss.spines['top'].set_visible(False)
ax_rosss.spines['right'].set_visible(False)
ax_rosss = fp.add_logticks(ax_rosss)
ax_rosss.tick_params(axis='x', which='major', pad=3)

ax_excursion = fig.add_subplot(gs[1, 0])
excursion(ax_excursion)

gs22 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1],
                                        hspace=0.1)
ax_spikecost = fig.add_subplot(gs22[1, 0])
ax_fakespike = fig.add_subplot(gs22[0, 0], sharex=ax_spikecost)
single_spike(ax_fakespike)
quantum(ax_spikecost)

ax_steadys = fig.add_subplot(gs[2, 0])
ax_steadys = figure_steady_state_simpler(ax_steadys)
ax_steadys = fp.add_logticks(ax_steadys)
ax_steadys.tick_params(axis='x', which='major', pad=3)

ax_compensate = fig.add_subplot(gs[2, 1])
ax_compensate = metabolic_spikes(ax_compensate)
ax_compensate.spines['top'].set_visible(False)
ax_compensate.spines['right'].set_visible(False)
ax_compensate = fp.add_logticks(ax_compensate)
ax_compensate.tick_params(axis='x', which='major', pad=3)

ax_fakespike.spines['top'].set_visible(False)
ax_fakespike.spines['right'].set_visible(False)
ax_fakespike.spines['bottom'].set_visible(False)
ax_fakespike.get_xaxis().set_visible(False)
ax_spikecost.spines['top'].set_visible(False)
ax_spikecost.spines['right'].set_visible(False)


# ax_rosland, ax_rosss
# ax_excursion, ax_fakespike
# ax_excursion, ax_spikecost
# ax_steadys, ax_compensate

fp.align_axis_labels([ax_steadys, ax_excursion,
                      ax_rosland],
                     axis='y', value=-0.15)
fp.align_axis_labels([ax_rosss, ax_fakespike,
                      ax_compensate],
                     axis='y', value=-0.15)
fp.align_axis_labels([ax_spikecost], axis='y', value=-0.02)

# fp.align_axis_labels([ax_rosland, ax_excursion,
#                      ax_compensate],
#                     axis='y', value=-0.15)

# fp.align_axis_labels([ax_steadys, ax_compensate],
#                      axis='x', value=-0.2)

fp.align_axis_labels([ax_rosland],
                     axis='x', value=-0.1)
fp.align_axis_labels([ax_spikecost],
                     axis='x', value=-0.28)
fp.align_axis_labels([ax_excursion],
                     axis='x', value=-0.14)

gs.tight_layout(fig)
rect = 0.125, 0.67, 0.33, 0.01
cbaxes = fig.add_axes(rect)
cb = plt.colorbar(surf, cax=cbaxes,
                  orientation='horizontal', ticks=[0, 1])
cb.set_label('ROS level (a.u.)', labelpad=-5)
plt.savefig('Figure1.png', dpi=300)
# plt.show()

