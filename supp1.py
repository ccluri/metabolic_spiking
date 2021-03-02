
import numpy as np
from mitochondria import Mito
from utils import Recorder, Q_nak
from lifcell import LIFCell
from figure_properties import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches
from matplotlib.ticker import FixedLocator
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from steady_state import figure_reaction_rate_nullclines, get_steady_state
from gates import get_ros, get_ros_fast, get_ros_slow, ros_inf
from gates import ros_tau, ros_tau_fast, ros_tau_slow
import matplotlib
matplotlib.use('Agg')

def single_spike(ax0):
    '''Dummy spike'''
    params = {'deltaga': 0, 'refrc': 10,
              'Q':1, 'init_atp':1, 'g_nap':1.25, 'g_cat':1}
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
    ax0.set_title('A.P. and Costs')
    return ax0


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


def spike_quanta(baseline_atp, q, f_mcu=0.1e-4, ros=get_ros(), tau_Q=100):
    '''Perturbation due to a spike'''
    dt = 0.01
    time = 500
    # factor = 0.1e-4
    tt = np.arange(0, time, dt)
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # state 4 - wait till we reach here
    ros.init_val(m.atp, m.psi)
    Q_val = Q_nak(tt, q, tau_Q=tau_Q)
    spike_val = np.zeros_like(tt)
    ros_vals = np.zeros_like(tt)
    t_start = 150
    spike_val[int(t_start/dt):] += Q_val[:len(spike_val[int(t_start/dt):])]
    rec_vars_list = ['atp', 'psi', 'nad', 'pyr']
    m_record = Recorder(m, rec_vars_list, time, dt)
    for ii, ti in enumerate(tt):
        try:
            m.update_vals(dt, atp_cost=spike_val[ii],
                          leak_cost=spike_val[ii]*f_mcu)
        except IndexError:
            m.update_vals(dt, leak_cost=0, atp_cost=0)
        ros.update_vals(dt, m.atp, m.psi,
                        spike_val[ii]+baseline_atp)
        ros_vals[ii] = ros.get_val()
        m_record.update(ii)
    return m_record, tt, ros_vals


def substrate_pert(ax, baseline_atp, q, f_mcu):
    m_record, tt, ros_vals = spike_quanta(baseline_atp, q, f_mcu)
    ax.plot(tt, m_record.out['atp'],
            c=def_colors['atp'],
            label=r'$ATP_M$', lw=1)
    ax.plot(tt, m_record.out['psi'],
            c=def_colors['psi'],
            label=r'$\Delta\psi_M$', lw=1)
    ax.plot(tt, m_record.out['nad'],
            c=def_colors['nad'],
            label=r'$NAD^{+}$', lw=1)
    ax.plot(tt, m_record.out['pyr'],
            c=def_colors['pyr'],
            label=r'$Pyr$', lw=1)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 1.5])
    ax.set_yticklabels([0, 1.5])
    # ax.yaxis.get_ticklocs(minor=True)
    minorLocator = FixedLocator([0.5, 1.0])
    ax.yaxis.set_minor_locator(minorLocator)
    return ax


def q_pert(ax, qs):
    dt = 0.01
    tt = np.arange(0, 500, dt)
    for Q in qs:
        Qval = np.zeros_like(tt)
        vals = Q_nak(tt, fact=Q)
        Qval[int(150/dt):] += vals[:len(Qval[int(150/dt):])]
        ax.plot(tt, Qval, lw=1)
    ax.set_ylim(-5, 55)
    ax.set_yticks([0, 50])
    ax.set_ylabel('per-spike expense')
    return ax


def q_pert_rs(ax, qs, baseline_atp=30):
    for Q in qs:
        m_record, tt, rs = spike_quanta(baseline_atp, q=Q, f_mcu=0.1e-4)
        ax.plot(m_record.out['atp'], m_record.out['psi'], lw=1)
    ax.set_xlim(0.5, 1)
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.75])
    ax.set_aspect('equal')
    return ax


def q_pert_sc(ax, qs):

    for Q in qs:
        b_test, vals = run_sim(test_freq=10, spike_quanta=Q)
        # ax.semilogx(b_test, vals, marker='o',
        #             lw=0.5, markersize=2)
        ax.semilogx(b_test, vals, lw=1,
                    label='Q='+str(Q))
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
    # ax.set_xlabel(r'$ATP_C \rightarrow ADP_C$ (/ms)')
    # ax.set_ylabel(r'ROS Signal (a.u.)')
    # ax.legend(frameon=False, handlelength=1,
    #           bbox_to_anchor=(0.2, .8, .6, .2),
    #           loc='lower center', ncol=2)
    return ax


def tau_q_pert(ax, tau_qs):
    dt = 0.01
    tt = np.arange(0, 500, dt)
    for tau in tau_qs:
        Qval = np.zeros_like(tt)
        vals = Q_nak(tt, fact=30, tau_Q=tau)
        Qval[int(150/dt):] += vals[:len(Qval[int(150/dt):])]
        ax.plot(tt, Qval, lw=1)
    ax.set_ylim(-5, 35)
    ax.set_ylabel('per-spike expense')
    ax.set_yticks([0, 30])
    return ax


def tau_q_pert_rs(ax, tau_qs, baseline_atp=30):
    for tau in tau_qs:
        m_record, tt, rs = spike_quanta(baseline_atp, q=30,
                                        f_mcu=0.1e-4, tau_Q=tau)
        ax.plot(m_record.out['atp'], m_record.out['psi'], lw=1)
    ax.set_xlim(0.5, 1)
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.75])
    ax.set_aspect('equal')
    return ax


def tau_q_pert_sc(ax, tau_qs):
    for tau_Q in tau_qs:
        b_test, vals = run_sim(test_freq=10, spike_quanta=30,
                               tau_Q=tau_Q)
        # ax.semilogx(b_test, vals, marker='o',
        #             lw=0.5, markersize=2)
        ax.semilogx(b_test, vals, lw=1,
                    label=r'$\tau_Q=$'+str(tau_Q)+'ms')
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
    return ax


def f_mcu_pert_rs(ax, f_mcus, baseline_atp=30):
    for f_mcu in f_mcus:
        m_record, tt, rs = spike_quanta(baseline_atp, q=30,
                                        f_mcu=f_mcu, tau_Q=100)
        ax.plot(m_record.out['atp'], m_record.out['psi'], lw=1)
    ax.set_xlim(0.5, 1)
    ax.set_xticks([0.5, 1])
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.75])
    ax.set_aspect('equal')
    return ax


def f_mcu_pert_sc(ax, f_mcus):
    for f_mcu in f_mcus:
        b_test, vals = run_sim(test_freq=10, spike_quanta=30,
                               psi_fac=f_mcu)
        # ax.semilogx(b_test, vals, marker='o',
        #             lw=0.5, markersize=2)
        ax.semilogx(b_test, vals, lw=1,
                    label=r'$f_{MCU}=$'+'{:.0e}'.format(f_mcu))
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
    return ax


def tau_ros_pert(ax, tau_ross):
    bls = np.geomspace(1, 1000, 100)
    for pp, tau_ros in enumerate(tau_ross):
        tau = np.zeros_like(bls)
        for ii, bl in enumerate(bls):
            tau[ii] = tau_ros(1, bl)
        ax.semilogx(bls, tau, lw=1)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 1500])
    ax.set_yticklabels([0, 1.5])
    ax.set_ylabel(r'$\tau_{ROS}$ (s)')
    # ax.set_ylim(-10, 35)
    # ax.set_aspect('equal')
    return ax


def tau_ros_pert_rs(ax, gate_ross, baseline_atp=30):
    for gate_ros in gate_ross:
        m_record, tt, rs = spike_quanta(baseline_atp, q=30,
                                        tau_Q=100, ros=gate_ros)
        ax.plot(tt, rs, lw=1)
    # ax.set_ylim(0, 0.5)
    ax.set_aspect(2000)
    ax.set_ylim([0, 0.3])
    ax.set_yticks([0, 0.3])
    ax.set_ylabel('ROS signal (au)')
    return ax


def tau_ros_pert_sc(ax, gate_ross):
    scavs = [0.1, 1, 10]
    for pp, gate_ros in enumerate(gate_ross):
        b_test, vals = run_sim(test_freq=10, spike_quanta=30,
                               ros=gate_ros)
        # ax.semilogx(b_test, vals, marker='o',
        #             lw=0.5, markersize=2)
        ax.semilogx(b_test, vals, lw=1,
                    label=r'$f_{SCAV}=$'+str(scavs[pp]))
    ax = add_ros_ss(ax)
    ax.set_xscale('log')
    ax.set_ylim(0., 1.)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([0, 0.5, 1])
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
                lw=1, label='Control',
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


figsize = cm_to_inches([8.9, 19])
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(6, 3, figure=fig, height_ratios=[1.8, 0.9, 1, 1, 1, 1],
                       width_ratios=[1, 1, 1], hspace=0.05, wspace=0.1)
# gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :],
#                                        width_ratios=[1, 1], wspace=0)
# gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[:, 1],
#                                         hspace=0.1)

# ax0 = fig.add_subplot(gs00[0, 0])
# ax0 = single_spike(ax0)
# ax1 = fig.add_subplot(gs00[1, 0])
# ax1 = quantum(ax1)
# neat_axes([ax0, ax1])

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :],
                                       width_ratios=[1, 1], wspace=0.2)
atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
ax11 = fig.add_subplot(gs1[0, 0])
ax11 = figure_reaction_rate_nullclines(baseline_atp=30,
                                       ax=ax11)
cntrlx, cntrly = atp(30), psi(30)
ax11.plot(cntrlx, cntrly, marker='*', markersize=7, color='k',
          markeredgecolor='none', markeredgewidth=.5)
rect1 = patches.Rectangle((0.5, 0.25), 0.5, 0.5,
                          edgecolor='gray', linewidth=0.5,
                          facecolor='none')
ax11.add_patch(rect1)

ax12 = fig.add_subplot(gs1[0, 1])
ax12 = figure_reaction_rate_nullclines(baseline_atp=150,
                                       ax=ax12)
cntrlx, cntrly = atp(150), psi(150)
ax12.plot(cntrlx, cntrly, marker='*', markersize=7, color='gold',
          markeredgecolor='k', markeredgewidth=.5)
# neat_axes([ax11, ax12])

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :],
                                       width_ratios=[1, 1], wspace=0.2)
ax21 = fig.add_subplot(gs2[0, 0])
ax21 = substrate_pert(ax21, baseline_atp=30, q=50, f_mcu=0.1e-4)
ax21.plot([150, 150], [0, 1.5], ls='--',
          lw=0.5, c='k', zorder=1.9)
ax21.plot(150, 1.4, marker='*', markersize=7, color='k',
          markeredgecolor='none', markeredgewidth=.5)
ax22 = fig.add_subplot(gs2[0, 1])
ax22 = substrate_pert(ax22, baseline_atp=150, q=50, f_mcu=0.1e-4)
ax22.plot([150, 150], [0, 1.5], ls='--',
          lw=0.5, c='k', zorder=1.9)
ax22.plot(150, 1.4, marker='*', markersize=7, color='gold',
          markeredgecolor='k', markeredgewidth=.5)
ax21.set_ylabel('Subst. conc. (au)')
add_sizebar(ax21, 100)
add_sizebar(ax22, 100)
strip_axes([ax21, ax22])

gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :],
                                       width_ratios=[1, 1, 1], wspace=0.2)
ax31 = fig.add_subplot(gs3[0, 0])
ax31 = q_pert(ax31, qs=[50, 30, 10])
ax32 = fig.add_subplot(gs3[0, 1])
ax32 = q_pert_rs(ax32, qs=[50, 30, 10], baseline_atp=30)
ax33 = fig.add_subplot(gs3[0, 2])
ax33 = q_pert_sc(ax33, qs=[50, 30, 10])
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
add_sizebar(ax41, 100, offset=-0.2)
strip_axes([ax41])
neat_axes([ax43])

gs5 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[4, :],
                                       width_ratios=[1, 1, 1], wspace=0.2)
ax52 = fig.add_subplot(gs5[0, 1])
ax52 = f_mcu_pert_rs(ax52, f_mcus=[0, 5e-5, 1e-4], baseline_atp=30)
ax53 = fig.add_subplot(gs5[0, 2])
ax53 = f_mcu_pert_sc(ax53, f_mcus=[0, 5e-5, 1e-4])
neat_axes([ax53])

gs6 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[5, :],
                                       width_ratios=[1, 1, 1], wspace=0.2)
ax61 = fig.add_subplot(gs6[0, 0])
ax61 = tau_ros_pert(ax61, [ros_tau_fast, ros_tau, ros_tau_slow])
ax62 = fig.add_subplot(gs6[0, 1])
ax62 = tau_ros_pert_rs(ax62, gate_ross=[get_ros_fast(), get_ros(),
                                        get_ros_slow()], baseline_atp=30)
ax63 = fig.add_subplot(gs6[0, 2])
ax63 = tau_ros_pert_sc(ax63, gate_ross=[get_ros_fast(),
                                        get_ros(),
                                        get_ros_slow()])
add_sizebar(ax62, 100, offset=-0.2)
strip_axes([ax62])
neat_axes([ax61, ax63])

ax21.legend(frameon=False, ncol=4, loc='upper left',
            bbox_to_anchor=(0.3, 1.3))
ax33.legend(frameon=False, ncol=4, loc='upper left',
            bbox_to_anchor=(-2.25, -0.2))
ax43.legend(frameon=False, ncol=4, loc='upper left',
            bbox_to_anchor=(-2.5, -0.2))
ax53.legend(frameon=False, ncol=4, loc='upper left',
            bbox_to_anchor=(-2.5, -0.2))
ax63.legend(frameon=False, ncol=4, loc='upper left',
            bbox_to_anchor=(-2.5, -0.2))

align_axis_labels([ax11, ax21, ax31, ax41, ax61], axis='y', value=-0.15)
align_axis_labels([ax62], axis='y', value=-0.1)
align_axis_labels([ax11, ax12], axis='x', value=-0.1)
ax31.set_title('Perturbation')
ax32.set_title('Resp. state space')
ax33.set_title('Spike compensation @10Hz')

gs.tight_layout(fig)
plt.savefig('supp1.png', dpi=300, transparent=True)
# plt.show()