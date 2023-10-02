import numpy as np

from steady_state import get_steady_state
from math import exp
from gates import ros_inf

import figure_properties as fp
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

delta_v = 65-40  # change in resting
delta_h = 0  # 3
delta_n = 10
delta_b = 0
delta_a = 0

n_alpha = lambda v: (0.01 * (v + 55 - delta_v - delta_n)) / (1 - exp(-0.1*(v+55-delta_v-delta_n)))
n_beta = lambda v: 0.125 * exp(-0.0125*(v+65-delta_v-delta_n))
n_inf = lambda v: n_alpha(v) / (n_alpha(v) + n_beta(v))
n_tau = lambda v: 3.25 / (n_alpha(v) + n_beta(v))

m_alpha = lambda v: (0.1 * (v+40-delta_v)) / (1-exp(-0.1*(v+40-delta_v)))
m_beta = lambda v: 4.0 * exp(-0.0556*(v+65-delta_v))
h_alpha = lambda v: 0.07 * exp(-0.05*(v+65-delta_v-delta_h))
h_beta = lambda v: 1.0 / (1 + exp(-0.1*(v+35-delta_v-delta_h)))
m_inf = lambda v: m_alpha(v) / (m_alpha(v) + m_beta(v))
m_tau = lambda v: 1 / (m_alpha(v) + m_beta(v))
h_inf = lambda v: h_alpha(v) / (h_alpha(v) + h_beta(v))
h_tau = lambda v: 1 / (h_alpha(v) + h_beta(v))

a_inf = lambda v: (1. / (1 + exp(-(v + 31-delta_v-delta_a) / 6.))) ** 0.25
a_tau = lambda v: ((100. / (7 * exp((v+60-delta_v-delta_a) / 14.) + 29 * exp(-(v+60-delta_v-delta_a) / 24.))) + 0.1)
b_inf = lambda v: 1. / (1 + exp((v + 66-delta_v-delta_b) / 7.)) ** 0.5
b_tau = lambda v: (0.25*(1000. / (14 * exp((v+60-delta_v-delta_b) / 27.) + 29 * exp(-(v+60-delta_v-delta_b) / 24.))) + 1)
c_inf = lambda v: 1. / (1 + exp((v + 66-delta_v-delta_b) / 7.)) ** 0.5
c_tau = lambda v: 35 # ((90. / (1 + exp((-66-v-delta_v-delta_b) / 17.))) + 10)

dadt = lambda v, a : (a_inf(v) - a) / a_tau(v)
dbdt = lambda v, b : (b_inf(v) - b) / b_tau(v)
dcdt = lambda v, c : (c_inf(v) - c) / c_tau(v)
dndt = lambda v, n : (n_inf(v) - n) / n_tau(v)
dmdt = lambda v, m : (m_inf(v) - m) / m_tau(v)
dhdt = lambda v, h : (h_inf(v) - h) / h_tau(v)


def I_KA(V, a, b, c, g_KA=60, f=0.7, E_K=-60.):
    return g_KA * a**4 * c * ((1-f)*b + f) * (V - E_K)


def I_KDR(V, n, g_KDR=80, E_K=-60):
    return g_KDR * n**4 * (V - E_K)


def I_NA(V, m, h, g_NA=1000, E_NA=40):
    return g_NA * m**3 * h * (V - E_NA)


def dvmdt(v, i_stim, iextra=0):
    dvdt = tau_m_inv * (v_leak - v + (R_m*i_stim) - iextra)
    return dvdt


def ros_ss(ax, ross, cases, atp_bl):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    these_ros = []
    for ros in ross:
        ros1_vals = np.zeros_like(bls)
        for ii, bl in enumerate(bls):
            ros1_vals[ii] = ros(atp(bl), psi(bl))
        these_ros.append(ros1_vals)
    case_clrs = ['k',
                 fp.def_colors['minisog'],
                 fp.def_colors['aox']]

    xy_leftbox = [1, 0]
    p = Rectangle(xy_leftbox, bls[48]-0.1, 1, clip_on=False,
                  edgecolor='none', facecolor='#dcdcdc', alpha=1)
    ax.add_patch(p)
    xy_leftbox = [bls[72], 0]
    p = Rectangle(xy_leftbox, bls[-1], 1, clip_on=False,
                  edgecolor='none', facecolor='#dcdcdc', alpha=1)
    ax.add_patch(p)
    
    ax.plot(bls[:49], these_ros[0][:49], lw=0.5, c='k', ls='-.')
    ax.plot(bls[72:], these_ros[0][72:], lw=0.5, c='k', ls='-.')
    # ax.plot(bls[:49], these_ros[0][:49], lw=0.7, c='gray', ls=(0, (1, 10)))
    # ax.plot(bls[72:], these_ros[0][72:], lw=0.7, c='gray', ls=(0, (1, 10)))
    ax.plot(bls[49:72], these_ros[0][49:72], label=cases[0], lw=0.5, c=case_clrs[0])
    ax.plot(bls[49:72], these_ros[0][49:72], label=' ', lw=0.5, c='white', alpha=0)
    ax.plot(bls[49:72], these_ros[1][49:72], label=cases[1], lw=0.5, c=case_clrs[1])
    ax.plot(bls[49:72], these_ros[2][49:72]-0.05, label=cases[2], lw=0.5, c=case_clrs[2])


    
    leg1 = plt.legend(bbox_to_anchor=(-0.4, 0.9, 0.4, 0.2),
                      bbox_transform=ax.transAxes,
                      frameon=False, loc='lower left',
                      ncol=2, handlelength=0.75, handletextpad=0.3)
    cnt_ros = ross[0](atp(atp_bl[0]), psi(atp_bl[0]))
    sog_ros = ross[1](atp(atp_bl[0]), psi(atp_bl[0]))
    aox_ros = ross[2](atp(atp_bl[0]), psi(atp_bl[0]))
    sd_ros = ross[0](atp(10), psi(10))
    
    # # draw vertical lines
    # ax.plot((atp_bl[0], atp_bl[0]), (0, sog_ros), c='gray', ls='--', lw=0.5)
    # ax.plot((10, 10), (0, sd_ros), c=fp.def_colors['SD'],
    #         ls='--', lw=0.5)
    # ax.plot((0, atp_bl[0]), (cnt_ros, cnt_ros), c='k', ls='--', lw=0.5)
    # ax.plot((0, atp_bl[0]), (sog_ros, sog_ros),
    #         c=fp.def_colors['minisog'], ls='--', lw=0.5)
    # ax.plot((0, atp_bl[0]), (cnt_ros, aox_ros),
    #         c=fp.def_colors['aox'], ls='--', lw=0.5)
    
    # ax.annotate('', (atp_bl[0], cnt_ros), xytext=(10, cnt_ros),
    #             textcoords='data', xycoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.01),
    #             horizontalalignment='right', verticalalignment='top')
    
    ax.plot(atp_bl[0], -0.1, marker='*', c='k', clip_on=False, markersize=7,
            markeredgewidth=0.5, markeredgecolor='none')
    # ax.plot(70, -0.1, marker='*', clip_on=False, markersize=7, c='gold',
    #         markeredgecolor='k', markeredgewidth=0.5, zorder=10)
    ax.plot(70, -0.1, marker='*', clip_on=False, markersize=7, c='#ff8c00',
            markeredgecolor='k', markeredgewidth=0.5, zorder=10)
    ax.plot(10, -0.1, marker='*', clip_on=False, markersize=7,
            c=fp.def_colors['SD'],
            markeredgecolor='k', markeredgewidth=0.5, zorder=10)

    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0., .5, 1])
    ax.set_xscale('log')
    ax.set_xlabel('Non-spiking costs (%s)' % kANT_units)
    ax = fp.add_logticks(ax)
    ax.tick_params(axis='x', which='major', pad=3)
    ax.set_ylabel(r'ROS level (a.u.)')
    plt.gca().add_artist(leg1)
    return ax


def voltage_trace(ax, ff_sleep=0.7, ff_awake=0):
    dt = 0.01
    t = np.arange(0, 50, dt)
    y = np.zeros_like(t)
    y += 30.
    y[:50] = -60
    aj = np.zeros_like(t)
    bj = np.zeros_like(t)
    cj = np.zeros_like(t)
    aj[0] = 0
    bj[0] = 1
    cj[0] = 1
    ikas_awake = np.zeros_like(t)
    ikas_sleep = np.zeros_like(t)
    for i in range(1, len(t)):  # Compute currents
        aj[i] = aj[i-1] + (dadt(y[i-1], aj[i-1]) * dt)
        bj[i] = bj[i-1] + (dbdt(y[i-1], bj[i-1]) * dt)
        cj[i] = cj[i-1] + (dcdt(y[i-1], cj[i-1]) * dt)
        ikas_awake[i] = I_KA(y[i-1], aj[i], bj[i], cj[i], f=ff_awake)
        ikas_sleep[i] = I_KA(y[i-1], aj[i], bj[i], cj[i], f=ff_sleep)
    norm_max = max(max(ikas_awake), max(ikas_sleep))
    # ax.plot(t, ikas_awake/norm_max, lw=0.5, color='gold', label='Inactivating')
    ax.plot(t, ikas_awake/norm_max, lw=0.5, color='#ff8c00', label='Inactivating')
    ax.plot(t, ikas_sleep/norm_max, lw=0.5, color='k',
            label='Non inactivating')
    ax.legend(frameon=False, loc=9, ncol=1, handlelength=0.7)
    ax.set_ylim([-0.1, 1.7])
    ymin, ymax = ax.get_ybound()
    asb = AnchoredSizeBar(ax.transData,
                          int(10),
                          '10 ms',
                          loc='lower center',
                          bbox_to_anchor=(0.8, -0.1),
                          bbox_transform=ax.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax.add_artist(asb)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Norm. current    ')
    ax.set_title('K Channel, A-type')
    return ax


def iclamp(tt, clamp, hold):
    if tt < 100:
        return hold
    elif tt < 1100:
        return clamp
    else:
        return hold


def current_clamp(ff, curr_clamps=[100]):
    dt = 0.01
    t = np.arange(0, 1600, dt)
    curr_clamps = curr_clamps
    k_currs_dict = {}
    gates_dict = {}
    spike_all = []
    mem_pots_dict = {}
    spike_times_dict = {}
    for clamp in curr_clamps:
        spike_times_dict[clamp] = []
        spike_count = 0
        y = np.zeros_like(t)
        aj = np.zeros_like(t)
        bj = np.zeros_like(t)
        cj = np.zeros_like(t)
        nj = np.zeros_like(t)
        mj = np.zeros_like(t)
        hj = np.zeros_like(t)
        y[0] = -60
        aj[0] = 0
        bj[0] = 1
        cj[0] = 1
        nj[0] = 0
        mj[0] = 0
        hj[0] = 1
        ikas = np.zeros_like(t)
        ikdrs = np.zeros_like(t)
        inas = np.zeros_like(t)
        refract = False
        for i in range(1, len(t)):  # Compute currents
            aj[i] = aj[i-1] + (dadt(y[i-1], aj[i-1]) * dt)
            bj[i] = bj[i-1] + (dbdt(y[i-1], bj[i-1]) * dt)
            cj[i] = cj[i-1] + (dcdt(y[i-1], cj[i-1]) * dt)
            nj[i] = nj[i-1] + (dndt(y[i-1], nj[i-1]) * dt)
            mj[i] = mj[i-1] + (dmdt(y[i-1], mj[i-1]) * dt)
            hj[i] = hj[i-1] + (dhdt(y[i-1], hj[i-1]) * dt)
            ikdr = I_KDR(y[i-1], nj[i], g_KDR=g_kdr)
            ina = I_NA(y[i-1], mj[i], hj[i], g_NA=g_na)
            ika = I_KA(y[i-1], aj[i], bj[i], cj[i], g_KA=g_ka, f=ff)
            ikdrs[i] = ikdr
            inas[i] = ina
            ikas[i] = ika
            # iextras[i] = iextra
            stim = iclamp(t[i], clamp, hold)
            y[i] = y[i-1] + (dvmdt(y[i-1], i_stim=stim,
                                   iextra=ina+ikdr+ika)*dt)
            if y[i] >= -20:
                if not refract:
                    spike_times_dict[clamp].append(t[i])
                    spike_count += 1
                    refract = True
            else:
                refract = False
        k_currs_dict[clamp] = [ikas, ikdrs]
        mem_pots_dict[clamp] = y
        gates_dict[clamp] = {'h': hj}
        spike_all.append(spike_count)
        print(spike_count, clamp)
    return t, mem_pots_dict, k_currs_dict, spike_times_dict, gates_dict


def plot_membpot(ax, ff, clamp, t, mem_pots_dict):
    y = mem_pots_dict[clamp]
    if ff == 0:
        label = 'Inactivating'
        # color = 'gold'
        color = '#ff8c00'
        axs = True
    else:
        label = 'Non inactivating'
        color = 'k'
        axs = False
    ax.plot(t, y, lw=0.5, color=color, label=label)
    ax.set_ylim([-70, 20])
    ax.get_xaxis().set_ticks([])
    if axs:
        ax.set_ylabel('Memb. pot. (mV)')
        ax.set_yticks(np.arange(-60, 20, 20))
    else:
        ax.set_yticks(np.arange(-60, 20, 20))
        ax.set_yticklabels([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    start_time = 150
    end_time = 400
    xy_inset = (start_time, -40)
    wd_inset = end_time - start_time
    ht_inset = 55
    p = Rectangle(xy_inset, wd_inset, ht_inset,
                  edgecolor='k', facecolor='none', lw=0.5, zorder=10)
    ax.add_patch(p)
    ymin, ymax = ax.get_ybound()
    asb = AnchoredSizeBar(ax.transData,
                          int(400),
                          '400 ms',
                          loc='lower center',
                          bbox_to_anchor=(0.37, 0.),
                          bbox_transform=ax.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax.add_artist(asb)
    return ax, start_time, end_time


def plot_area(ax, ff, clamp, t, k_currs_dict, gates_dict,
              start_time=180, end_time=380, leg=False):
    ika, ikdr = k_currs_dict[clamp]
    gates = gates_dict[clamp]
    itot = np.array(ika) + np.array(ikdr)
    imax = np.max(itot)
    lns = []
    lns.append(ax.fill_between(t, np.array(ikdr)/imax,
                               (np.array(ikdr)+np.array(ika))/imax,
                               color='#9d65d0', label='A-type'))
    lns.append(ax.fill_between(t, np.zeros_like(t),
                               np.array(ikdr)/imax, color='#d4bde9',
                               label='Non A-type'))
    if gates is not None:
        for gate in gates:
            lns.append(ax.plot(t, gates[gate], lw=0.5,
                               label=gate, c='k'))
    if leg:
        ax.legend(frameon=False, loc=9, ncol=3,
                  handlelength=0.7,
                  bbox_to_anchor=(-0.05, 1))
    ax.spines['left'].set_visible(False)
    ax.set_ylim(0., 1.4)
    ax.set_xlim(start_time, end_time)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([200, 300])
    ax.set_xlabel('Time (ms)')
    return ax


def add_connections(ax, ax1, start_time, end_time):
    xy_inset = (start_time, -40)
    wd_inset = end_time - start_time
    ccp = ConnectionPatch(xyA=xy_inset, xyB=(0, 1), axesA=ax, axesB=ax1,
                          coordsA="data", coordsB="axes fraction", lw=0.3)
    ax.add_patch(ccp)
    ccp2 = ConnectionPatch(xyA=(xy_inset[0] + wd_inset, xy_inset[1]),
                           xyB=(1, 1), axesA=ax, axesB=ax1,
                           coordsA="data", coordsB="axes fraction", lw=0.3)
    ax.add_patch(ccp2)
    return ax, ax1


def clean_ax(axs):
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)

kANT_units = '10$^{-3}$/s'
hold = -20  # pA
v_leak = -40
tau_m = 100.  # ms
tau_m_inv = 1 / tau_m
R_m = 1   # Gohm

# kA : 60, DR: 80, Na:1200; 15:23
g_ka = 80
g_kdr = 90
g_na = 1200

figsize = fp.cm_to_inches([8.9, 10])
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(3, 3, wspace=0.05, hspace=0.05)
gs00 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[:, 1:],
                                        wspace=0.2, hspace=0.4,
                                        height_ratios=[1, 1, 1])

ax11 = plt.subplot(gs[1, 0])
ax11 = ros_ss(ax11,
              [ros_inf,
               fp.dfb['ros_sog'],
               fp.dfb['ros_aox']],
              ['Control', 'miniSOG', 'AOX'],
              fp.dfb['atp_bl'])

ax1 = plt.subplot(gs00[0, 1])
ax2 = plt.subplot(gs00[1, 0])
ax3 = plt.subplot(gs00[1, 1])
ax4 = plt.subplot(gs00[2, 0])
ax5 = plt.subplot(gs00[2, 1])

ff_sleep = 0.7
ff_awake = 0.

ax1 = voltage_trace(ax1, ff_sleep, ff_awake)
curr_clamps = [20]
test_clamp = 20

print('awake, F=', ff_awake)
t, mem_pots_dict, k_currs_dict, spike_times_dict, gates_dict = current_clamp(ff_awake,
                                                                             curr_clamps)
ax2, start_time, end_time = plot_membpot(ax2, ff_awake, test_clamp, t, mem_pots_dict)
ax4 = plot_area(ax4, ff_awake, test_clamp, t, k_currs_dict, gates_dict,
                start_time=start_time, end_time=end_time)
ax2, ax4 = add_connections(ax2, ax4, start_time, end_time)
ax4.set_ylabel('Norm. current (a.u.)')

print('sleep, F=', ff_sleep)
t, mem_pots_dict, k_currs_dict, spike_times_dict, gates_dict = current_clamp(ff_sleep,
                                                                             curr_clamps)
ax3, start_time, end_time = plot_membpot(ax3, ff_sleep, test_clamp, t, mem_pots_dict)
ax5 = plot_area(ax5, ff_sleep, test_clamp, t, k_currs_dict, gates_dict,
                start_time=start_time, end_time=end_time, leg=True)
ax3, ax5 = add_connections(ax3, ax5, start_time, end_time)
ax5.set_yticklabels([])

clean_ax([ax1, ax2, ax3, ax4, ax5, ax11])
plt.text(0.7, 0.65, s='Current clamp (20pA)',
         horizontalalignment='center',
         verticalalignment='center',
         bbox=dict(boxstyle="round",
                   ec=(1., 1., 1.),
                   fc=(1., 1., 1.)),
         transform=fig.transFigure, color='k')
plt.text(0.7, 0.35, s='K currents',
         horizontalalignment='center',
         verticalalignment='center',
         bbox=dict(boxstyle="round",
                   ec=(1., 1., 1.),
                   fc=(1., 1., 1.)),
         transform=fig.transFigure, color='k')

gs.tight_layout(fig)
plt.savefig('Figure3_dfb.png', dpi=300)
# plt.show()
