import numpy as np

# from utils import Recorder, Q_nak
# from mitochondria import Mito
from mitosfns import spike_quanta, run_sim
from steady_state import get_steady_state

from gates import get_ros, get_ros_fast, get_ros_slow, ros_inf
from gates import ros_tau, ros_tau_fast, ros_tau_slow
from gates import get_ros_const, get_ros_const_fast, get_ros_const_slow
from gates import ros_tau_const, ros_tau_const_fast, ros_tau_const_slow

import figure_properties as fp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def tau_ros_pert(ax, tau_ross, cm=0):
    if cm == 0:
        cmap = fp.ln_cols_ros
        scavs = [0.1, 1, 10]
    else:
        cmap = fp.ln_cols_ros2
        scavs = [5, 50, 500]
    bls = np.geomspace(1, 1000, 100)
    for pp, tau_ros in enumerate(tau_ross):
        tau = np.zeros_like(bls)
        for ii, bl in enumerate(bls):
            tau[ii] = tau_ros(1, bl)
        if cm == 0:
            ax.loglog(bls, tau, lw=1.5, c=cmap[pp],
                      label=str(scavs[pp]))
        else:
            ax.loglog(bls, tau, lw=1.5, c=cmap[pp],
                      label=str(scavs[pp]))
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_ylabel(r'$\tau_{ROS}$ (ms)')
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax = fp.add_logticks(ax, xax=False, ticks=[0.1, 1, 10, 100, 1000])
    ax.set_ylim([0.01, 10000])
    ax = fp.add_logticks(ax)
    return ax


def tau_ros_pert_rs(ax, gate_ross, baseline_atp=30, cm=0):
    if cm == 0:
        cmap = fp.ln_cols_ros
        scavs = [0.1, 1, 10]
    else:
        cmap = fp.ln_cols_ros2
        scavs = [5, 50, 500]
    for gate_ros, col, ss in zip(gate_ross, cmap, scavs):
        m_record, tt, rs = spike_quanta(baseline_atp, q=0.1,
                                        tau_Q=100, ros=gate_ros)
        ax.plot(tt, rs, lw=1.5, c=col, label=str(ss))
    ax.set_ylim([0, 0.3])
    ax.set_yticks([0, 0.3])
    ax.plot([150, 150], [0, 0.3], '--', color='k', lw=0.5, zorder=1.9)
    ax.text(180, 0.035, s='Spike', transform=ax.transData)
    ax.set_ylabel('ROS (a.u.)')
    return ax


def tau_ros_pert_sc(ax, gate_ross, freq=10, cm=0):
    if cm == 0:
        cmap = fp.ln_cols_ros
    else:
        cmap = fp.ln_cols_ros2
    # scavs = [0.1, 1, 10]
    for pp, gate_ros in enumerate(gate_ross):
        b_test, vals = run_sim(test_freq=freq, spike_quanta=0.1,
                               ros=gate_ros)
        ax.semilogx(b_test, vals, lw=1.5, c=cmap[pp])
        # ax.semilogx(b_test, vals, lw=1.5, c=cmap[pp],
        #             label=r'$f_{SCAV}=$'+str(scavs[pp])+'ms')
    ax = add_ros_ss(ax)
    ax.set_ylabel('ROS (a.u.)')
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
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
                lw=1, label='Control (0Hz)',
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
                          bbox_to_anchor=(0.8+offset, 0.1),
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
gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1],
                       width_ratios=[1, 1])  # , hspace=.5, wspace=0.1)

ax00 = fig.add_subplot(gs[0, 0])
ax00 = tau_ros_pert(ax00, [ros_tau_fast, ros_tau, ros_tau_slow])
ax10 = fig.add_subplot(gs[1, 0])
ax10 = tau_ros_pert_rs(ax10, gate_ross=[get_ros_fast(),
                                        get_ros(),
                                        get_ros_slow()], baseline_atp=30)
ax20 = fig.add_subplot(gs[2, 0])  # 5 Hz
ax20 = tau_ros_pert_sc(ax20, gate_ross=[get_ros_fast(),
                                        get_ros(),
                                        get_ros_slow()],
                       freq=5)
ax30 = fig.add_subplot(gs[3, 0])  # 10 Hz
ax30 = tau_ros_pert_sc(ax30, gate_ross=[get_ros_fast(),
                                        get_ros(),
                                        get_ros_slow()],
                       freq=10)
ax40 = fig.add_subplot(gs[4, 0])  # 20 Hz
ax40 = tau_ros_pert_sc(ax40, gate_ross=[get_ros_fast(),
                                        get_ros(),
                                        get_ros_slow()],
                       freq=20)
add_sizebar(ax10, 200, offset=-0.2)
strip_axes([ax10])
neat_axes([ax00, ax20, ax30, ax40])


ax01 = fig.add_subplot(gs[0, 1])
ax01 = tau_ros_pert(ax01, [ros_tau_const_fast,
                           ros_tau_const,
                           ros_tau_const_slow], cm=1)
ax11 = fig.add_subplot(gs[1, 1])
ax11 = tau_ros_pert_rs(ax11, gate_ross=[get_ros_const_fast(),
                                        get_ros_const(),
                                        get_ros_const_slow()],
                       baseline_atp=30, cm=1)
ax21 = fig.add_subplot(gs[2, 1])  # 5 Hz
ax21 = tau_ros_pert_sc(ax21, gate_ross=[get_ros_const_fast(),
                                        get_ros_const(),
                                        get_ros_const_slow()],
                       freq=5, cm=1)
ax31 = fig.add_subplot(gs[3, 1])  # 10 Hz
ax31 = tau_ros_pert_sc(ax31, gate_ross=[get_ros_const_fast(),
                                        get_ros_const(),
                                        get_ros_const_slow()],
                       freq=10, cm=1)
ax41 = fig.add_subplot(gs[4, 1])  # 20 Hz
ax41 = tau_ros_pert_sc(ax41, gate_ross=[get_ros_const_fast(),
                                        get_ros_const(),
                                        get_ros_const_slow()],
                       freq=20, cm=1)
add_sizebar(ax11, 200, offset=-0.2)
strip_axes([ax11])
neat_axes([ax01, ax21, ax31, ax41])

align_axis_labels([ax00, ax10, ax20, ax30, ax40], axis='y', value=-0.15)
align_axis_labels([ax01, ax11, ax21, ax31, ax41], axis='y', value=-0.15)

# ax20.legend(frameon=False, ncol=4, loc='upper left', handlelength=0.5,
#             bbox_to_anchor=(-2.25, -0.2))

ax01.set_ylim(ax00.get_ylim())
leg = ax00.legend(frameon=False, ncol=3, loc='lower left', handlelength=1,
                  mode="expand", title=r'$f_{SCAV}=$',
                  bbox_to_anchor=(0., 0.9, 1, 0.1), borderaxespad=0.)
leg._legend_box.align = "left"
leg = ax01.legend(frameon=False, ncol=3, loc='lower left', handlelength=1,
                  mode="expand", title=r'$\tau_{ROS}(ms)=$',
                  bbox_to_anchor=(0., 0.9, 1, 0.1))
leg._legend_box.align = "left"
leg = ax10.legend(frameon=False, ncol=3, loc='upper left', handlelength=1,
                  mode="expand", title=r'$f_{SCAV}=$',
                  bbox_to_anchor=(0., -0.1, 1, 0.1), borderaxespad=0.)
leg._legend_box.align = "left"
leg = ax11.legend(frameon=False, ncol=3, loc='upper left', handlelength=1,
                  mode="expand", title=r'$\tau_{ROS}(ms)=$',
                  bbox_to_anchor=(0., -0.1, 1, 0.1))
leg._legend_box.align = "left"

ax20.legend(frameon=False, ncol=1, loc='upper center',
            handlelength=0.5)
ax30.legend(frameon=False, ncol=1, loc='upper center',
            handlelength=0.5)
ax40.legend(frameon=False, ncol=1, loc='upper center',
            handlelength=0.5)
ax21.legend(frameon=False, ncol=1, loc='upper center',
            handlelength=0.5)
ax31.legend(frameon=False, ncol=1, loc='upper center',
            handlelength=0.5)
ax41.legend(frameon=False, ncol=1, loc='upper center',
            handlelength=0.5)

for axx in [ax00, ax01, ax20, ax21, ax30, ax31, ax40, ax41]:
    ii = axx.set_xlabel(r'Non-spiking costs (%s)' % Kant_units)
gs.tight_layout(fig)  # this is a workaround for formatting sake
for axx in [ax01, ax21, ax31, ax41]:
    ii = axx.set_xlabel('')
for axx in [ax00, ax20, ax30, ax40]:
    axx.xaxis.set_label_coords(1.2, -0.15)


ax10.text(1.2, 1.1, s='RETROS relief from one spike (    =30/ks)',
          transform=ax10.transAxes,
          va='center', ha='center', clip_on=False)
ax10.plot(1.61, 1.1, marker='*', markersize=7, color='k',
          markeredgecolor='k', markeredgewidth=.5,
          transform=ax10.transAxes, clip_on=False)
ax20.text(1.2, 1.1, s='Spike compensation @ 5Hz',
          transform=ax20.transAxes,
          va='center', ha='center', clip_on=False)
ax30.text(1.2, 1.1, s='Spike compensation @ 10Hz',
          transform=ax30.transAxes,
          va='center', ha='center', clip_on=False)
ax40.text(1.2, 1.1, s='Spike compensation @ 20Hz',
          transform=ax40.transAxes,
          va='center', ha='center', clip_on=False)

plt.savefig('Figure1_supp_ros_scav.png', dpi=300, transparent=False)
# plt.show()
