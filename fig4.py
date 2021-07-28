import numpy as np

from sncda import SNCDA
from gates import get_ros, ros_inf
from gates import get_parkinsons_rotenone
from gates import get_parkinsons_type2
from mitochondria import Mito
from utils import Recorder, Q_nak
from steady_state import get_steady_state

import figure_properties as fp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def run_sim(mito_baseline, spike_quanta, ros, time, dt, i_inj=None,
            Q_nak_def=Q_nak, psi_fac=0.1e-4):
    print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
    t = np.arange(0, time, dt)
    if i_inj is None:
        i_inj = np.zeros_like(t)
    else:
        assert(len(t) == len(i_inj))
    qdur = 2000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak_def(qtime, spike_quanta)
    qlen = len(this_q)
    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=1000)
    ros.init_val(1, 0)
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
    # params1 = {'Q': spike_quanta,
    #            'init_ros': ros.val, 'init_atp': mi.atp,
    #            'g_nap': 10, 'g_katp': 15}
    params1 = {'Q': spike_quanta,
               'init_ros': ros.val, 'init_atp': mi.atp,
               'g_nap': 110, 'g_katp': 100, 'g_leak': 15, 'C_m': 100}
    
    c1 = SNCDA('PD_model', **params1)
    r_cell = Recorder(c1, ['v', 'ros', 'atp'], time, dt)
    spike_expns = np.zeros_like(t)
    spikes = []
    for i in range(len(t)):
        c1.i_inj = i_inj[i]
        mi.update_vals(dt,
                       atp_cost=spike_expns[i],
                       leak_cost=spike_expns[i]*psi_fac)
        ros.update_vals(dt, mi.atp, mi.psi, spike_expns[i]+mito_baseline)
        c1.update_redox(mi.atp, ros.val)
        spk = c1.update_vals(dt)
        if spk:
            print('SPIKE!')
            spikes.append(t[i])
            try:
                spike_expns[i:i+qlen] += this_q
            except ValueError:
                spike_expns[i:] += this_q[:len(spike_expns[i:])]
        r_cell.update(i)
        r_mito.update(i)
    return r_cell, r_mito, spikes, t


def upper_run(axs, axs_meta, mito_baseline, spike_quanta):
    ax1, ax2, ax3, ax4 = axs
    ax1_m, ax2_m, ax3_m, ax4_m = axs_meta
    ros = get_ros()
    dt = 0.01
    time = 4000
    t = np.arange(0, time, dt)
    i_inj = np.zeros_like(t)
    t_start = 1000
    t_end = 1100
    i_inj[int(t_start/dt):int(t_end/dt)] = 300
    t_start = 2500
    t_end = 2800
    i_inj[int(t_start/dt):int(t_end/dt)] = 300
    t_start = 3800
    t_end = 3900
    i_inj[int(t_start/dt):int(t_end/dt)] = 300
    r_cell_cntrl, r_mito_cntrl, spikes_cntrl, t = run_sim(mito_baseline,
                                                          spike_quanta,
                                                          ros, time, dt, i_inj)
    park1 = get_parkinsons_rotenone()
    r_cell_typ1, r_mito_typ1, spikes_typ1, t = run_sim(mito_baseline,
                                                       spike_quanta,
                                                       park1, time, dt, i_inj)
    park2 = get_parkinsons_type2()
    r_cell_typ2, r_mito_typ2, spikes_typ2, t = run_sim(mito_baseline,
                                                       spike_quanta*3,
                                                       park2, time, dt, i_inj,
                                                       Q_nak_def=Q_nak,
                                                       psi_fac=(0.1e-4)/3)
    r_cell_typ3, r_mito_typ3, spikes_typ3, t = run_sim(90, spike_quanta,
                                                       ros, time, dt, i_inj)

    ax1.plot(t, r_cell_cntrl.out['v'], c='k', lw=0.25)
    ax2.plot(t, r_cell_typ1.out['v'],
             c=fp.def_colors['park1'], lw=0.25)
    ax3.plot(t, r_cell_typ2.out['v'],
             c=fp.def_colors['park2'], lw=0.25)
    ax4.plot(t, r_cell_typ3.out['v'],
             c='gold', lw=0.35)

    ax1_m.plot(t, r_cell_cntrl.out['atp'],
               c=fp.def_colors['atp'], lw=0.5, label='ATP')
    ax2_m.plot(t, r_cell_typ1.out['atp'],
               c=fp.def_colors['atp'], lw=0.5,  label='ATP')
    ax3_m.plot(t, r_cell_typ2.out['atp'],
               c=fp.def_colors['atp'], lw=0.5, label='ATP')
    ax4_m.plot(t, r_cell_typ3.out['atp'],
               c=fp.def_colors['atp'], lw=0.5, label='ATP')

    ax1_m.plot(t, r_cell_cntrl.out['ros'],
               c=fp.def_colors['ros'], lw=0.5, label='ROS')
    ax2_m.plot(t, r_cell_typ1.out['ros'],
               c=fp.def_colors['ros'], lw=0.5, label='ROS')
    ax3_m.plot(t, r_cell_typ2.out['ros'],
               c=fp.def_colors['ros'], lw=0.5, label='ROS')
    ax4_m.plot(t, r_cell_typ3.out['ros'],
               c=fp.def_colors['ros'], lw=0.5, label='ROS')

    add_sizebars([ax1_m, ax2_m, ax3_m, ax4_m])

    ax1_m.legend(frameon=False, loc='center', handlelength=1,
                 bbox_to_anchor=(0.5, -0.2), bbox_transform=ax1_m.transAxes,
                 ncol=2, framealpha=1)
    ax2_m.legend(frameon=False, loc='center', handlelength=1,
                 bbox_to_anchor=(0.5, -0.2), bbox_transform=ax2_m.transAxes,
                 ncol=2, framealpha=1)
    ax3_m.legend(frameon=False, loc='center', handlelength=1,
                 bbox_to_anchor=(0.5, -0.2), bbox_transform=ax3_m.transAxes,
                 ncol=2, framealpha=1)
    ax4_m.legend(frameon=False, loc='center', handlelength=1,
                 bbox_to_anchor=(0.5, -0.2), bbox_transform=ax4_m.transAxes,
                 ncol=2, framealpha=1)
    ax1.set_ylim(-80, 50)
    ax2.set_ylim(-80, 50)
    ax3.set_ylim(-80, 50)
    ax4.set_ylim(-80, 50)
    ax1.set_yticks([-60, 0])
    ax2.set_yticks([-60, 0])
    ax3.set_yticks([-60, 0])
    ax4.set_yticks([-60, 0])
    ax1_m.set_ylim(0, 1)
    ax2_m.set_ylim(0, 1)
    ax3_m.set_ylim(0, 1)
    ax4_m.set_ylim(0, 1)
    ax1.text(-100, 50, '(mV)', va='center', ha='left')
    ax2.text(-100, 50, '(mV)', va='center', ha='left')
    ax3.text(-100, 50, '(mV)', va='center', ha='left')
    ax4.text(-100, 50, '(mV)', va='center', ha='left')
    ax1.set_ylabel('Memb.\nPot.')
    ax2.set_ylabel('Memb.\nPot.')
    ax3.set_ylabel('Memb.\nPot.')
    ax4.set_ylabel('Memb.\nPot.')
    ax1_m.set_ylabel('(a.u.)')
    ax2_m.set_ylabel('(a.u.)')
    ax3_m.set_ylabel('(a.u.)')
    ax4_m.set_ylabel('(a.u.)')
    neat_axes([ax1, ax2, ax3, ax4, ax1_m, ax2_m, ax3_m, ax4_m])
    return t, i_inj


def neat_axes(axs):
    for ax in axs:
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return axs


def add_sizebars(axs, y_offset=-0.3):
    for ax in axs:
        ymin, ymax = ax.get_ybound()
        asb = AnchoredSizeBar(ax.transData,
                              int(200),
                              '200 ms',
                              loc='lower left',
                              bbox_to_anchor=(0.9, y_offset),
                              bbox_transform=ax.transAxes,
                              pad=0., borderpad=.0, sep=2,
                              frameon=False, label_top=False,
                              size_vertical=(ymax-ymin)/1000)
        ax.add_artist(asb)
    return


def ros_sss(ax, ross, cases, atp_bl):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    these_ros = []
    for ros in ross:
        ros1_vals = np.zeros_like(bls)
        for ii, bl in enumerate(bls):
            ros1_vals[ii] = ros(atp(bl), psi(bl))
        these_ros.append(ros1_vals)
    case_clrs = ['k', fp.def_colors['park1']]
    for ii, rr in enumerate(these_ros):
        ax.plot(bls, rr, label=cases[ii], lw=0.5,
                c=case_clrs[ii])
    cnt_ros = ross[0](atp(atp_bl[0]), psi(atp_bl[0]))
    park_ros = ross[1](atp(atp_bl[0]), psi(atp_bl[0]))
    ax.plot(atp_bl[0], -0.1, marker='*', c='k',
            clip_on=False, markersize=7,
            markeredgecolor='none')
    ax.plot(60, -0.1, marker='*', c='gold', label='Lysosomal',
            clip_on=False, markersize=7, zorder=10, linestyle='None',
            markeredgecolor='k', markeredgewidth=0.5)
    
    ax.set_ylim(-0.1, 1.)
    ax.set_yticks([0., 0.5, 1.])
    ax.set_yticklabels(['0', '', '1'])
    ax.set_xscale('log')
    ax = fp.add_logticks(ax)
    ax.set_xlabel('Non-spiking costs %s' % Kant_units)
    ax.set_ylabel('ROS level (a.u.)')
    return ax, cnt_ros, park_ros


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)
    return


def quantum(ax1):
    '''Illustrating baseline plus Q atp->adp'''
    dt = 0.01
    tt = np.arange(0, 500, dt)
    Qval = np.zeros_like(tt)
    vals = Q_nak(tt, 30)
    Qval[int(150/dt):] += vals[:len(Qval[int(150/dt):])]
    Qval_PD = np.zeros_like(tt)
    vals_PD = Q_nak(tt, 90)
    Qval_PD[int(150/dt):] += vals_PD[:len(Qval_PD[int(150/dt):])]
    ax1.plot(tt, Qval, lw=0.5, c='k', label='Control', zorder=2)
    ax1.plot(tt, Qval_PD, lw=0.5, c=fp.def_colors['park2'],
             label='Demyelination', zorder=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel(r'$ATP_C \rightarrow ADP_C$'+'\n%s' % Kant_units,
                   labelpad=7)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.plot(-25, 0, marker='*', c='k', clip_on=False, markersize=7,
             markeredgecolor='none')

    ax1.text(s='+Q', x=85, y=27.5, fontsize=5)
    ax1.text(s='+3Q', x=85, y=87.5, fontsize=5)
    ax1.set_xlim(-25, 500)
    ax1.set_ylim(-10, 120)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    return ax1


if __name__ == '__main__':
    Kant_units = '(10$^{-3}$/s)'
    figsize = fp.cm_to_inches([8.9, 12])
    fig = plt.figure(figsize=figsize)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.5, height_ratios=[1, 4])
    ax1 = plt.subplot(gs[0, 0])  # north west
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1, cnt_ros, park_ros = ros_sss(ax1, [ros_inf, fp.snc['ros']],
                                     ['Control', 'Rotenone'],
                                     [fp.snc['atp_bl']])
    ax2 = plt.subplot(gs[0, 1])
    ax2 = quantum(ax2)  # north east
    hndles, lbels = ax1.get_legend_handles_labels()
    hndles2, lbels2 = ax2.get_legend_handles_labels()
    hndles.append(hndles2[-1])
    lbels.append(lbels2[-1])
    leg1 = plt.legend(hndles, lbels, frameon=False, loc='upper center',
                      ncol=4, bbox_to_anchor=(0, 0.816, 1, 0.2),
                      bbox_transform=fig.transFigure, handlelength=1)
    plt.gca().add_artist(leg1)
    align_axis_labels([ax1, ax2], axis='x', value=-0.2)
    gs00 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[1, :],
                                            hspace=0.3,
                                            height_ratios=[0.1, 1, 1, 1, 1])

    gs000 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[1, :],
                                             hspace=0.15,
                                             height_ratios=[1, 1])
    gs001 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[2, :],
                                             hspace=0.15,
                                             height_ratios=[1, 1])
    gs002 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[3, :],
                                             hspace=0.15,
                                             height_ratios=[1, 1])
    gs003 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[4, :],
                                             hspace=0.15,
                                             height_ratios=[1, 1])
    
    ax3 = plt.subplot(gs000[0, 0])
    ax3_meta = plt.subplot(gs000[1, 0])
    ax4 = plt.subplot(gs001[0, 0])
    ax4_meta = plt.subplot(gs001[1, 0])
    ax5 = plt.subplot(gs002[0, 0])
    ax5_meta = plt.subplot(gs002[1, 0])
    ax6 = plt.subplot(gs003[0, 0])
    ax6_meta = plt.subplot(gs003[1, 0])

    ax_iclamp = plt.subplot(gs00[0, :])
    axs = [ax3, ax4, ax5, ax6]
    axs_meta = [ax3_meta, ax4_meta, ax5_meta, ax6_meta]
    t, i_inj = upper_run(axs, axs_meta,
                         mito_baseline=fp.snc['atp_bl'],
                         spike_quanta=fp.snc['Q'])

    ax_iclamp.plot(t, i_inj, c='gray', lw=0.5)
    ax_iclamp.spines['top'].set_visible(False)
    ax_iclamp.spines['right'].set_visible(False)
    ax_iclamp.spines['bottom'].set_visible(False)
    ax_iclamp.spines['left'].set_visible(False)
    ax_iclamp.get_xaxis().set_visible(False)
    ax_iclamp.get_yaxis().set_visible(False)
    add_sizebars([ax_iclamp], y_offset=-2)
    ax_iclamp.set_xlabel('Time (ms)')
    ax_iclamp.set_ylabel('Current\nclamp')
    ax_iclamp.text(-100, 300, '(nA)', va='center', ha='left')
    align_axis_labels([ax3, ax3_meta, ax4, ax4_meta, ax5, ax5_meta, ax_iclamp,
                       ax6, ax6_meta],
                      axis='y', value=-0.07)
    align_axis_labels([ax1], axis='y', value=-0.2)
    gs.tight_layout(fig)
    plt.savefig('Figure4.png', dpi=300)
