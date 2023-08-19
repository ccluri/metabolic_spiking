import numpy as np

from utils import Q_nak
from mitosfns import spike_quanta, run_sim
from steady_state import get_steady_state
from gates import ros_inf

import figure_properties as fp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def substrate_pert(ax, baseline_atp, q):
    m_record, tt, ros_vals = spike_quanta(baseline_atp, q)
    ax.plot(tt, m_record.out['atp'],
            c=fp.def_colors['atp'],
            label=r'$ATP_M$', lw=1.5)
    ax.plot(tt, m_record.out['psi'],
            c=fp.def_colors['psi'],
            label=r'$\Delta\psi$', lw=1.5)
    ax.plot(tt, m_record.out['nad'],
            c=fp.def_colors['nad'],
            label=r'NAD+', lw=1.5)
    ax.plot(tt, m_record.out['pyr'],
            c=fp.def_colors['pyr'],
            label=r'Pyruvate', lw=1.5)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 1.5])
    ax.set_yticklabels([0, 1.5])
    minorLocator = FixedLocator([0.5, 1.0])
    ax.yaxis.set_minor_locator(minorLocator)
    return ax


def figure_reaction_rate_nullclines(baseline_atp, ax):
    filename = 'reaction_rates/reaction_rates_baseline_'
    filename += str(baseline_atp)+'.npz'
    print(filename)
    out = np.load(filename)
    ATPx = out['ATPx']
    PSIx = out['PSIx']
    U = out['DATPDT']
    V = out['DPSIDT']
    if baseline_atp == 30:
        manual_positiona = None
        manual_positiond = [(0.5, 0.45)]
    elif baseline_atp == 150:
        manual_positiona = [(0.5, 0.5)]
        manual_positiond = [(0.75, 0.3)]
    strm_atp = ax.contour(ATPx, PSIx, U.T, levels=[0],
                          colors=[fp.def_colors['atp']], linewidths=1,
                          linestyles='dashdot')
    strm_psi = ax.contour(ATPx, PSIx, V.T, levels=[0],
                          colors=[fp.def_colors['psi']], linewidths=1,
                          linestyles='dashed')
    ax.clabel(strm_atp, inline=True, fontsize=7,
              fmt=r'$\frac{dATP_M}{dt}=0$', manual=manual_positiona)
    ax.clabel(strm_psi, inline=True, fontsize=7,
              fmt=r'$\frac{d\Delta\psi}{dt}=0$', manual=manual_positiond)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_aspect('equal')
    ax.set_xlabel('$ATP_M$')
    ax.set_ylabel('$\Delta\psi$')
    return ax


def q_pert(ax, qs):
    dt = 0.01
    tt = np.arange(0, 500, dt)
    for Q, col in zip(qs, fp.ln_cols_q):
        Qval = np.zeros_like(tt) + 0.7
        vals = Q_nak(tt, fact=Q, tau_rise=5)
        Qval[int(150/dt):] -= vals[:len(Qval[int(150/dt):])]
        ax.plot(tt, Qval, lw=1.5, c=col)
    ax.set_ylim(0.4, 1.1)
    ax.set_yticks([0.5, 1])
    ax.set_ylabel('Per-spike cost Q\n(a.u.)')
    return ax


def q_pert_rs(ax, qs, baseline_atp=30):
    for Q, col in zip(qs, fp.ln_cols_q):
        m_record, tt, rs = spike_quanta(baseline_atp, q=Q)
        ax.plot(m_record.out['atp'], m_record.out['psi'], lw=1.5, c=col)
    ax.set_xlim(0.5, 1)
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.75])
    ax.set_aspect('equal')
    return ax


def q_pert_sc(ax, qs):
    for Q, col in zip(qs, fp.ln_cols_q):
        b_test, vals = run_sim(test_freq=10, spike_quanta=Q)
        ax.semilogx(b_test, vals, lw=1.5,
                    label='Q='+str(Q), c=col)
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
    ax = fp.add_logticks(ax)
    return ax


def tau_q_pert(ax, tau_qs):
    dt = 0.01
    tt = np.arange(0, 500, dt)
    for tau, col in zip(tau_qs, fp.ln_cols_tau):
        Qval = np.zeros_like(tt) + 0.7
        vals = Q_nak(tt, fact=0.1, tau_Q=tau, tau_rise=5)
        Qval[int(150/dt):] -= vals[:len(Qval[int(150/dt):])]
        ax.plot(tt, Qval, lw=1.5, c=col)
    ax.set_ylim(0.4, 1.1)
    ax.set_ylabel('Per-spike cost Q\n(a.u.)')
    ax.set_yticks([0.5, 1])
    return ax


def tau_q_pert_rs(ax, tau_qs, baseline_atp=30):
    for tau, col in zip(tau_qs, fp.ln_cols_tau):
        m_record, tt, rs = spike_quanta(baseline_atp, q=0.1,
                                        f_mcu=0.1e-4, tau_Q=tau)
        ax.plot(m_record.out['atp'], m_record.out['psi'], lw=1.5, c=col)
    ax.set_xlim(0.5, 1)
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.75])
    ax.set_aspect('equal')
    return ax


def tau_q_pert_sc(ax, tau_qs):
    for tau_Q, col in zip(tau_qs, fp.ln_cols_tau):
        b_test, vals = run_sim(test_freq=10, spike_quanta=0.1,
                               tau_Q=tau_Q)
        ax.semilogx(b_test, vals, lw=1.5, c=col,
                    label=r'$\tau_{Qfall}=$'+str(tau_Q)+'ms')
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
    ax = fp.add_logticks(ax)
    return ax


def f_mcu_pert_rs(ax, f_mcus, baseline_atp=30):
    for f_mcu, col in zip(f_mcus, fp.ln_cols_mcu):
        m_record, tt, rs = spike_quanta(baseline_atp, q=0.1,
                                        f_mcu=f_mcu, tau_Q=100)
        ax.plot(m_record.out['atp'], m_record.out['psi'], lw=1.5, c=col)
    ax.set_xlim(0.5, 1)
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.75])
    ax.set_aspect('equal')
    return ax


def f_mcu_pert_sc(ax, f_mcus):
    for f_mcu, col in zip(f_mcus, fp.ln_cols_mcu):
        b_test, vals = run_sim(test_freq=10, spike_quanta=0.1,
                               psi_fac=f_mcu)
        if f_mcu == 0:
            label = r'$f_{MCU}=0$'
        else:
            label = r'$f_{MCU}=$'+str(round(f_mcu*1000, 2))+'M/V'
        ax.semilogx(b_test, vals, lw=1.5, c=col,
                    label=label)
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
    ax = fp.add_logticks(ax)
    return ax


def tau_ros_pert(ax, tau_ross):
    bls = np.geomspace(1, 1000, 100)
    for pp, tau_ros in enumerate(tau_ross):
        tau = np.zeros_like(bls)
        for ii, bl in enumerate(bls):
            tau[ii] = tau_ros(1, bl)
        ax.loglog(bls, tau, lw=1.5, c=fp.ln_cols_ros[pp])
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_ylabel(r'$\tau_{ROS}$ (ms)')
    ax.set_yticks([1, 10, 100, 1000])
    ax = fp.add_logticks(ax)
    return ax


def neat_axes(axs):
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return axs


def strip_axes(axs, both=False):
    axs = neat_axes(axs)
    for ax in axs:
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        if both:
            ax.spines['left'].selfet_visible(False)
            ax.get_yaxis().set_visible(False)
    return axs


def add_ros_ss(ax):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    ros_vals = np.zeros_like(bls)
    for ii, bl in enumerate(bls):
        ros_vals[ii] = ros_inf(atp(bl), psi(bl))
    ax.semilogx(bls, ros_vals,
                lw=1, label='Ctrl',
                c='k', zorder=1.9)
    return ax


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)


def add_sizebar(ax, val, offset=0):
    ymin, ymax = ax.get_ybound()
    asb = AnchoredSizeBar(ax.transData,
                          int(val),
                          str(val)+' ms',
                          loc='lower left',
                          bbox_to_anchor=(0.8+offset, -0.15),
                          bbox_transform=ax.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax.add_artist(asb)
    return ax


Kant_units = '10$^{-3}$/s'
figsize = fp.cm_to_inches([8.9, 20])
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(6, 3, figure=fig, height_ratios=[1.8, 0.9, 1, 1, 1, 1],
                       width_ratios=[1, 1, 1], hspace=0.05, wspace=0.1)

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :],
                                       width_ratios=[1, 1], wspace=0.2)
atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
ax11 = fig.add_subplot(gs1[0, 0])
ax11 = figure_reaction_rate_nullclines(baseline_atp=30,
                                       ax=ax11)
cntrlx, cntrly = atp(30), psi(30)
ax11.plot(cntrlx, cntrly, marker='*', markersize=7, color='k',
          markeredgecolor='k', markeredgewidth=.5)
rect1 = patches.Rectangle((0.5, 0.25), 0.5, 0.5,
                          edgecolor='gray', linewidth=0.5,
                          facecolor='none')
ax11.add_patch(rect1)

ax12 = fig.add_subplot(gs1[0, 1])
ax12 = figure_reaction_rate_nullclines(baseline_atp=150,
                                       ax=ax12)
cntrlx, cntrly = atp(150), psi(150)
ax12.plot(cntrlx, cntrly, marker='*', markersize=7, color='gold',
          markeredgecolor='k', markeredgewidth=.5, zorder=10)

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :],
                                       width_ratios=[1, 1], wspace=0.2)
ax21 = fig.add_subplot(gs2[0, 0])
ax21 = substrate_pert(ax21, baseline_atp=30, q=0.1)
ax21.plot([150, 150], [0, 1.5], ls='--',
          lw=0.5, c='k', zorder=1.9)
ax21.plot(150, 1.4, marker='*', markersize=7, color='k',
          markeredgecolor='k', markeredgewidth=.5, zorder=10, clip_on=False)
ax22 = fig.add_subplot(gs2[0, 1])
ax22 = substrate_pert(ax22, baseline_atp=150, q=0.1)
ax22.plot([150, 150], [0, 1.5], ls='--',
          lw=0.5, c='k', zorder=1.9)
ax22.plot(150, 1.4, marker='*', markersize=7, color='gold', clip_on=False,
          markeredgecolor='k', markeredgewidth=.5)
ax21.set_ylabel('Subst. conc.\n(a.u.)')
add_sizebar(ax21, 100)
add_sizebar(ax22, 100)
strip_axes([ax21, ax22])

gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :],
                                       width_ratios=[1, 1, 1], wspace=0.2)
ax31 = fig.add_subplot(gs3[0, 0])
ax31 = q_pert(ax31, qs=[0.2, 0.1, 0.05])
ax32 = fig.add_subplot(gs3[0, 1])
ax32 = q_pert_rs(ax32, qs=[0.15, 0.1, 0.05], baseline_atp=30)
ax33 = fig.add_subplot(gs3[0, 2])
ax33 = q_pert_sc(ax33, qs=[0.15, 0.1, 0.05])
cntrlx, cntrly = atp(30), psi(30)
ax32.plot(cntrlx, cntrly, marker='*', markersize=7, color='k',
          markeredgecolor='None', zorder=-1)
add_sizebar(ax31, 100, offset=-0.2)
strip_axes([ax31])
neat_axes([ax33])

gs4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, :],
                                       width_ratios=[1, 1, 1], wspace=0.2)
ax41 = fig.add_subplot(gs4[0, 0])
ax41 = tau_q_pert(ax41, tau_qs=[300, 100, 25])
ax42 = fig.add_subplot(gs4[0, 1])
ax42 = tau_q_pert_rs(ax42, tau_qs=[300, 100, 25], baseline_atp=30)
ax43 = fig.add_subplot(gs4[0, 2])
ax43 = tau_q_pert_sc(ax43, tau_qs=[300, 100, 25])
cntrlx, cntrly = atp(30), psi(30)
ax42.plot(cntrlx, cntrly, marker='*', markersize=7, color='k',
          markeredgecolor='None', zorder=-1)
add_sizebar(ax41, 100, offset=-0.2)
strip_axes([ax41])
neat_axes([ax43])

gs5 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[4, :],
                                       width_ratios=[1, 1, 1], wspace=0.2)
ax52 = fig.add_subplot(gs5[0, 1])
ax52 = f_mcu_pert_rs(ax52, f_mcus=[0, 5e-3, 1e-2], baseline_atp=30)
ax53 = fig.add_subplot(gs5[0, 2])
ax53 = f_mcu_pert_sc(ax53, f_mcus=[0, 5e-3, 1e-2])
cntrlx, cntrly = atp(30), psi(30)
ax52.plot(cntrlx, cntrly, marker='*', markersize=7, color='k',
          markeredgecolor='None', zorder=-1)
neat_axes([ax53])

ax21.legend(frameon=False, ncol=4, loc='upper left', handlelength=0.5,
            bbox_to_anchor=(0.2, 1.5), )
ax33.legend(frameon=False, ncol=4, loc='upper left', handlelength=0.5,
            bbox_to_anchor=(-2.25, -0.2))
ax43.legend(frameon=False, ncol=4, loc='upper left', handlelength=0.5,
            bbox_to_anchor=(-2.9, -0.2))
ax53.legend(frameon=False, ncol=4, loc='upper left', handlelength=0.5,
            bbox_to_anchor=(-2.8, -0.2))

align_axis_labels([ax11], axis='y', value=-0.15)
align_axis_labels([ax21, ax31, ax41], axis='y', value=-0.1)
align_axis_labels([ax11, ax12], axis='x', value=-0.1)
ax31.set_title('Perturbation', pad=5)
ax32.set_title('Respiratory \nstate space',  pad=5)
ax33.set_title('Spike compensation\n@10Hz',  pad=5)
 
gs.tight_layout(fig)
plt.savefig('Figure1_supp.png', dpi=300, transparent=False)
# plt.show()
