import numpy as np

from mitochondria import Mito
from gates import get_ros, ros_inf
from utils import Recorder, Q_nak
from steady_state import get_steady_state

import figure_properties as fp
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches, colors, colorbar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from palettable.colorbrewer.qualitative import Set3_4_r


def fetch_isi(iclamp):
    E = -70  # mV
    Rm = 10  # Mohm
    tau = 10  # sec
    vr = -80  # mV
    vt = -54  # mV
    if Rm*iclamp <= vt-E:
        print('Too small current to ellicit spiking')
        isi = np.nan
    else:
        isi = lambda i: (tau * np.log((Rm*i + E - vr)/(Rm*i + E - vt)))
        return isi(iclamp)


def fetch_rel_spikes(t, spk, start, end, arr):
    relevant_spikes = spk[np.where((spk >= start) &
                                   (spk <= end))]
    arr_val_at_spike = []
    for sp in relevant_spikes:
        arr_val_at_spike.append(arr[np.where(t == sp)[0][0]])
    renorm_spikes = [(dd-start)*100 for dd in relevant_spikes]
    return renorm_spikes, arr_val_at_spike


def fetch_steadystate_ros(bls):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    atp_ss = []
    ros_ss = []
    for bl in bls:
        ros_ss.append(ros_inf(atp(bl), psi(bl)))
        atp_ss.append(atp(bl))
    return ros_ss, atp_ss


def print_ax_lims(axs):
    for ax in axs:
        print(ax.get_xlim(), ax.get_ylim())


def ret_run_sim(case, theta_ret=0.025):
    mito_baseline = case.bl
    spike_quanta = case.q
    psi_fac = case.psi_fac
    refrac = case.refrac
    print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
    print('Psi factor: ', psi_fac, 'Refrac :', refrac)
    dt = 0.01
    time = 2500
    t = np.arange(0, time, dt)
    qdur = 1000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, fact=1, tau_rise=5)
    qlen = len(this_q)
    ros = get_ros()
    # Mitochondria
    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=1000)
    ros.init_val(mi.atp, mi.psi)
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
    spike_expns = np.zeros_like(t) + mi.atp
    leak_expns = np.zeros_like(t)
    ros_vals = np.zeros_like(t)
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
        ros_vals[i] = ros.get_val()
        msig = ros.get_val()*(mi.atp - 0.71218258)
        ms_vals[i] = msig
        if msig > theta_ret:  # RET ROS
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
    print(spikes)
    return spikes, t, r_mito, ros_vals, ms_vals


def fet_run_sim(case, iclamp=2, theta_fet=-0.025):
    mito_baseline = case.bl
    spike_quanta = case.q
    psi_fac = case.psi_fac
    refrac = case.refrac
    print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
    print('Psi factor: ', psi_fac, 'Refrac :', refrac, 'I clamp :', iclamp)
    dt = 0.01
    time = 2500
    t = np.arange(0, time, dt)
    #  I clamp
    i_stim = np.zeros_like(t)
    stim_start = 700
    stim_end = 1000
    i_stim[int(stim_start/dt):int(stim_end/dt)] = iclamp  # nA
    isi_min = fetch_isi(iclamp)
    #  Spike costs
    qdur = 1000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, fact=1, tau_rise=5)
    qlen = len(this_q)
    #  Mitochondria
    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=1000)
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
    #  ROS states
    ros = get_ros()
    ros.init_val(mi.atp, mi.psi)
    # Init vars
    spike_expns = np.zeros_like(t) + mi.atp
    leak_expns = np.zeros_like(t)
    ros_vals = np.zeros_like(t)
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
        ros_vals[i] = ros.get_val()
        msig = ros.get_val()*(mi.atp - 0.71218258)
        ms_vals[i] = msig
        if i_stim[i] > 0 and not np.isnan(isi_min):  # FET ROS
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
            if elapsed < max(isi_min, refrac) or msig < theta_fet:
                elapsed += dt
            else:
                spiked = False
        r_mito.update(i)
    print(spikes)
    return spikes, t, r_mito, ros_vals, ms_vals, i_stim


def make_raster_plot(ax, ii, spk, spikes_offset, l_col):
    length = 2
    if np.min(np.diff(spk)) < 3:
        lw = 0.5
    else:
        lw = 0.7
    for sp in spk:
        ax.plot([sp-spikes_offset, sp-spikes_offset],
                [ii, ii+length], lw=lw, c=l_col)
    return ax


def ret_cases_considered(gs0, firing_tests, theta_ret=0.025):
    ax0 = plt.subplot(gs0[0, 0])  # i clamp  and spike raster
    ax1 = plt.subplot(gs0[1, 0])  # ROS inset
    ax2 = plt.subplot(gs0[2, 0])  # ATP levels
    ax3 = plt.subplot(gs0[3, 0])  # MS fet
    spikes_offset = 0
    l_cols = []
    for ii, case in enumerate(firing_tests):
        spk, t, r_mito, ros_vals, ms_vals = ret_run_sim(case,
                                                        theta_ret=theta_ret)
        atp_vals = r_mito.out['atp']
        if case.start is False:
            ax1.plot(t, ros_vals)
            ax2.plot(t, atp_vals)
            ax3.plot(t, ms_vals)
            ros_at_spike = []
            atp_at_spike = []
            ms_at_spike = []
            for sp in spk:
                ros_at_spike.append(ros_vals[np.where(t == sp)[0][0]])
                atp_at_spike.append(atp_vals[np.where(t == sp)[0][0]])
                ms_at_spike.append(ms_vals[np.where(t == sp)[0][0]])
            ax1.plot(spk, ros_at_spike, 'ko')
            ax2.plot(spk, atp_at_spike, 'ko')
            ax3.plot(spk, ms_at_spike, 'ko')
        else:
            start_idx = np.where(np.isclose(t, case.start))[0][0]
            end_idx = np.where(np.isclose(t, case.end))[0][0]
            l_curr = ax1.plot(ros_vals[start_idx:end_idx], zorder=-ii,
                              lw=0.5, color=fp.ln_cols_ret[ii])
            l_col = l_curr[0].get_color()
            l_cols.append(l_col)
            spk = np.array(spk)
            renorm_spikes, ros_at_spike = fetch_rel_spikes(t, spk, case.start,
                                                           case.end, ros_vals)
            ax1.plot(renorm_spikes, ros_at_spike, marker='o',
                     c='k', lw=0, zorder=-ii,
                     markersize=2,  markerfacecolor=l_col, markeredgewidth=0.2)
            renorm_spikes, atp_at_spike = fetch_rel_spikes(t, spk, case.start,
                                                           case.end, atp_vals)
            ax2.plot(atp_vals[start_idx:end_idx], lw=0.5,
                     color=fp.ln_cols_ret[ii], zorder=-ii)
            ax2.plot(renorm_spikes, atp_at_spike, marker='o',
                     c='k', lw=0, zorder=-ii,
                     markersize=2,  markerfacecolor=l_col, markeredgewidth=0.2)
            renorm_spikes, ms_at_spike = fetch_rel_spikes(t, spk, case.start,
                                                          case.end, ms_vals)
            ax3.plot(ms_vals[start_idx:end_idx], lw=0.5,
                     color=fp.ln_cols_ret[ii], zorder=-ii)
            ax3.plot(renorm_spikes, ms_at_spike, marker='o',
                     c='k', lw=0, zorder=-ii,
                     markersize=2,  markerfacecolor=l_col, markeredgewidth=0.2)
            spikes_offset = case.start - firing_tests[-1].start
            make_raster_plot(ax0, ii*3, spk, spikes_offset, l_col)
    #  Set limits
    if case.start is not False:
        ax0.set_xlim(case.start-25, 900)
        # ax0.set_xlim(0, end_idx-start_idx)
        ax1.set_xlim(0, end_idx-start_idx)
        ax2.set_xlim(0, end_idx-start_idx)
        ax3.set_xlim(0, end_idx-start_idx)

        ax1.set_ylim(0.07, 0.2)
        ax2.set_ylim(0.8, 0.9)
        ax3.set_ylim(0.005, 0.027)
        ax3.plot([0, 30000], [theta_ret, theta_ret], lw=0.5, ls='--',
                 color=fp.def_colors['ret'], zorder=-11)
        ax3.text(10000, 0.025, s=r'$\theta_{RET}$', ha='right',
                 color=fp.def_colors['ret'], va='bottom',
                 transform=ax3.transData, fontsize=7, clip_on=False)
        # # Ticks
        ax0.set_yticks([])
        ax0.spines['left'].set_visible(False)
        ax1.set_yticks([0.05, 0.18])
        ax2.set_yticks([0.8, 0.9])
        ax3.set_yticks([0.005, 0.033])
        ax1.ticklabel_format(axis='y',
                             style='sci', scilimits=(0, 0))
        ax2.ticklabel_format(axis='y',
                             style='sci', scilimits=(0, 0))
        ax3.ticklabel_format(axis='y',
                             style='sci', scilimits=(0, 0))

        ax1.yaxis.get_offset_text().set_visible(False)
        ax2.yaxis.get_offset_text().set_visible(False)
        ax3.yaxis.get_offset_text().set_visible(False)
        # print(tx, 'yabadabadoo')
        fontprops = fm.FontProperties(size=6)
        ax1.text(0., 1., s="1e-1", ha='left', va='center',
                 transform=ax1.transAxes, fontproperties=fontprops)
        ax2.text(0., 1., s="1e-1", ha='left', va='center',
                 transform=ax2.transAxes, fontproperties=fontprops)
        ax3.text(0., 1., s="1e-2", ha='left', va='center',
                 transform=ax3.transAxes, fontproperties=fontprops)
        # Labels
        ax0.set_ylabel('Spikes')
        ax1.set_ylabel('ROS\n(a.u.)')
        ax2.set_ylabel('ATP\n(a.u.)')
        ax3.set_ylabel('MS \n(a.u.)')

        # Add rectangles
        xy_inset = (case.start, -1)
        wd_inset = case.end - case.start
        ht_inset = 10.5
        rect = patches.Rectangle(xy_inset, wd_inset, ht_inset,
                                 linewidth=0.3,
                                 edgecolor='gray', facecolor='none')
        ax0.add_patch(rect)
        ccp = patches.ConnectionPatch(xyA=xy_inset, xyB=(0, 1), axesA=ax0,
                                      axesB=ax1, lw=0.3, color='gray',
                                      coordsA="data", coordsB="axes fraction")
        ax0.add_patch(ccp)
        ccp2 = patches.ConnectionPatch(xyA=(xy_inset[0] + wd_inset,
                                            xy_inset[1]), xyB=(1, 1),
                                       axesA=ax0, axesB=ax1, lw=0.3,
                                       color='gray',
                                       coordsA="data", coordsB="axes fraction")
        ax0.add_patch(ccp2)
        
    # Add sizebars
    ymin, ymax = ax0.get_ybound()

    asb0 = AnchoredSizeBar(ax0.transData,
                           int(50),
                           '50 ms',
                           bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax0.transAxes,
                           loc=3, fontproperties=fontprops,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    ax0.add_artist(asb0)

    ymin, ymax = ax1.get_ybound()
    asb1 = AnchoredSizeBar(ax1.transData,
                           int(10*100),
                           '10 ms', fontproperties=fontprops,
                           bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax1.transAxes,
                           loc=3,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    ax1.add_artist(asb1)

    ymin, ymax = ax2.get_ybound()
    asb2 = AnchoredSizeBar(ax2.transData,
                           int(10*100),
                           '10 ms', bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax2.transAxes,
                           loc=3, fontproperties=fontprops,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    ax2.add_artist(asb2)

    ymin, ymax = ax3.get_ybound()
    asb3 = AnchoredSizeBar(ax3.transData,
                           int(10*100),
                           '10 ms', bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax3.transAxes,
                           loc=3, fontproperties=fontprops,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    ax3.add_artist(asb3)
    print('RET')
    print_ax_lims([ax0, ax1, ax2, ax3])
    align_axis_labels([ax0, ax1, ax2, ax3], axis='y', value=-0.07)
    clean_ax([ax0, ax1, ax2, ax3])


def fet_cases_considered(gs0, firing_tests, theta_fet):
    ax0 = plt.subplot(gs0[0, 0])  # i clamp  and spike raster
    ax1 = plt.subplot(gs0[1, 0])  # ROS inset
    ax2 = plt.subplot(gs0[2, 0])  # ATP levels
    ax3 = plt.subplot(gs0[3, 0])  # MS fet
    l_cols = []
    for ii, case in enumerate(firing_tests):
        spk, t, r_mito, ros_vals, ms_vals, i_stim = fet_run_sim(case,
                                                                iclamp=2,
                                                                theta_fet=theta_fet)
        atp_vals = r_mito.out['atp']
        if ii == 0:
            ax0.plot(t, np.array(i_stim) + 6, 'k', lw=0.5)
        if case.start is False:
            ax0.plot(spk, [ii*1.5]*len(spk), 'ko', )
            l_curr = ax1.plot(t, ros_vals)
            l_cols.append(l_curr[0].get_color())
            ax2.plot(t, r_mito.out['atp'])
            ros_at_spike = []
            atp_at_spike = []
            for sp in spk:
                ros_at_spike.append(ros_vals[np.where(t == sp)[0][0]])
                atp_at_spike.append(atp_vals[np.where(t == sp)[0][0]])
            ax1.plot(spk, ros_at_spike, 'ko')
            ax2.plot(spk, atp_at_spike, 'ko')
        else:
            start_idx = np.where(np.isclose(t, case.start))[0][0]
            end_idx = np.where(np.isclose(t, case.end))[0][0]
            l_curr = ax1.plot(ros_vals[start_idx:end_idx], lw=0.5,
                              color=fp.ln_cols_fet[ii])
            ax2.plot(atp_vals[start_idx:end_idx], lw=0.5,
                     color=fp.ln_cols_fet[ii])
            ax3.plot(ms_vals[start_idx:end_idx], lw=0.5,
                     color=fp.ln_cols_fet[ii])
            l_col = l_curr[0].get_color()
            l_cols.append(l_col)
            spk = np.array(spk)
            renorm_spikes, ros_at_spike = fetch_rel_spikes(t, spk, case.start,
                                                           case.end, ros_vals)
            ax1.plot(renorm_spikes, ros_at_spike, marker='o', c='k',
                     lw=0, markerfacecolor=l_col,
                     markersize=2, markeredgewidth=0.2)
            renorm_spikes, atp_at_spike = fetch_rel_spikes(t, spk, case.start,
                                                           case.end, atp_vals)
            ax2.plot(renorm_spikes, atp_at_spike, marker='o', c='k',
                     lw=0, markerfacecolor=l_col,
                     markersize=2, markeredgewidth=0.2)
            renorm_spikes, ms_at_spike = fetch_rel_spikes(t, spk, case.start,
                                                          case.end, ms_vals)
            ax3.plot(renorm_spikes, ms_at_spike, marker='o', c='k',
                     lw=0, markerfacecolor=l_col,
                     markersize=2, markeredgewidth=0.2)
            make_raster_plot(ax0, ii*3, spk, spikes_offset=0, l_col=l_col)

    #  Set limits
    ax0.set_xlim(300, 1500)
    ax1.set_xlim(0, end_idx-start_idx)
    ax2.set_xlim(0, end_idx-start_idx)
    ax3.set_xlim(0, end_idx-start_idx)
    ax3.plot([0, 79999], [theta_fet, theta_fet], lw=0.5, ls='--',
             color=fp.def_colors['fet'], zorder=-11)
    ax3.text(55000, theta_fet, s=r'$\theta_{FET}$', ha='right',
             color=fp.def_colors['fet'], va='top',
             transform=ax3.transData, fontsize=7, clip_on=False)
    ax1.set_ylim(0.07, 0.2)
    ax2.set_ylim(0.1, 0.7)
    ax3.set_ylim(-0.07, 0.01)
    # # Ticks
    ax1.set_yticks([0.07, 0.2])
    ax2.set_yticks([.1, .7])
    ax3.set_yticks([-0.07, .01])
    ax1.ticklabel_format(axis='y',
                         style='sci', scilimits=(0, 0))
    ax2.ticklabel_format(axis='y',
                         style='sci', scilimits=(0, 0))
    ax3.ticklabel_format(axis='y',
                         style='sci', scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_visible(False)
    ax2.yaxis.get_offset_text().set_visible(False)
    ax3.yaxis.get_offset_text().set_visible(False)
    fontprops = fm.FontProperties(size=6)
    ax1.text(0., 1., s="1e-1", ha='left', va='center',
             transform=ax1.transAxes, fontproperties=fontprops)
    ax2.text(0., 1., s="1e-1", ha='left', va='center',
             transform=ax2.transAxes, fontproperties=fontprops)
    ax3.text(0., 1., s="1e-2", ha='left', va='center',
             transform=ax3.transAxes, fontproperties=fontprops)
    ax0.set_yticks([])
    ax0.spines['left'].set_visible(False)
    # Labels
    ax0.text(0.05, 0.9, s='Current \nclamp', transform=ax0.transAxes,
             ha='left', va='center', color='k', zorder=-1, clip_on=False)
    ax0.set_ylabel('Spikes')
    ax1.set_ylabel('ROS\n(a.u.)')
    ax2.set_ylabel('ATP\n(a.u.)')
    ax3.set_ylabel('MS \n(a.u.)')
    # Add rectangles
    xy_inset = (case.start, -1)
    wd_inset = case.end - case.start
    ht_inset = 10.5
    rect = patches.Rectangle(xy_inset, wd_inset, ht_inset,
                             linewidth=0.3,
                             edgecolor='gray', facecolor='none')
    ax0.add_patch(rect)
    ccp = patches.ConnectionPatch(xyA=xy_inset, xyB=(0, 1), axesA=ax0,
                                  axesB=ax1, lw=0.3, color='gray',
                                  coordsA="data", coordsB="axes fraction")
    ax0.add_patch(ccp)
    ccp2 = patches.ConnectionPatch(xyA=(xy_inset[0] + wd_inset,
                                        xy_inset[1]), xyB=(1, 1),
                                   axesA=ax0, axesB=ax1, lw=0.3, color='gray',
                                   coordsA="data", coordsB="axes fraction")
    ax0.add_patch(ccp2)

    # Add sizebars
    ymin, ymax = ax0.get_ybound()

    asb0 = AnchoredSizeBar(ax0.transData,
                           int(100),
                           '100 ms', fontproperties=fontprops,
                           loc=3, bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax0.transAxes,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    asb0.set_clip_on(False)
    ax0.add_artist(asb0)

    ymin, ymax = ax1.get_ybound()
    asb1 = AnchoredSizeBar(ax1.transData,
                           int(50*100),
                           '50 ms', fontproperties=fontprops,
                           loc=3, bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax1.transAxes,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    asb1.set_clip_on(False)
    ax1.add_artist(asb1)

    ymin, ymax = ax2.get_ybound()
    asb2 = AnchoredSizeBar(ax2.transData,
                           int(50*100),
                           '50 ms', fontproperties=fontprops,
                           loc=3, bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax2.transAxes,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    asb2.set_clip_on(False)
    ax2.add_artist(asb2)

    ymin, ymax = ax3.get_ybound()
    asb3 = AnchoredSizeBar(ax3.transData,
                           int(50*100),
                           '50 ms', fontproperties=fontprops,
                           loc=3, bbox_to_anchor=(0.8, -0.1),
                           bbox_transform=ax3.transAxes,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=False,
                           size_vertical=(ymax-ymin)/1000)
    asb3.set_clip_on(False)
    ax3.add_artist(asb3)

    print('FET')
    print_ax_lims([ax0, ax1, ax2, ax3])
    align_axis_labels([ax0, ax1, ax2, ax3], axis='y', value=-0.07)
    clean_ax([ax0, ax1, ax2, ax3])


def metabolic_signal_plot(gs, theta_ret, theta_fet):
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    bls = np.geomspace(1, 1000, 1000)
    ros_vals, atp_vals = fetch_steadystate_ros(bls)
    ros_vals = np.array(ros_vals)
    atp_vals = np.array(atp_vals)
    K_idx = np.where(ros_vals == np.min(ros_vals))
    atp_at_min = atp_vals[K_idx]
    ms = ros_vals*(atp_vals-atp_at_min)
    lm = ax0.semilogx(bls, ros_vals, lw=0.5, c='k', label='ROS')
    ax00 = ax0.twinx()
    ll = ax00.semilogx(bls, atp_vals-atp_at_min, lw=1,
                       color=fp.def_colors['atp'], label='$\partial$ATP')
    ax00x, ax00y = ax00.get_xlim()
    ax00.spines['right'].set_color(ll[0].get_color())
    ax00.tick_params(axis='y', colors=ll[0].get_color())
    ax00.set_ylabel('$\partial$ATP', labelpad=-22)
    ax00.yaxis.label.set_color(ll[0].get_color())
    
    ax1.semilogx(bls, ms, lw=0.5,
                 color='k', label='MS=ROS x $\partial$ATP')
    ax1.legend(frameon=True, handlelength=1, loc=9, borderaxespad=0,
               facecolor='white', framealpha=.0, edgecolor='white')
    ax1.plot([10, 100], [theta_ret, theta_ret], lw=0.5,
             ls='--', color=fp.def_colors['ret'], zorder=-1)
    ax1.plot([50, 500], [theta_fet, theta_fet], lw=0.5,
             ls='--', color=fp.def_colors['fet'], zorder=-1)
    ax1.text(500, theta_ret, s=r'$\theta_{RET}$', ha='right',
             color=fp.def_colors['ret'], va='center',
             transform=ax1.transData, fontsize=7, clip_on=False)
    ax1.text(30, theta_fet, s=r'$\theta_{FET}$', ha='right',
             color=fp.def_colors['fet'], va='center',
             transform=ax1.transData, fontsize=7, clip_on=False)
    # ax0.plot([bls[K_idx], bls[K_idx]],
    #          [0, atp_at_min], ls='--', lw=0.5,
    #          color='gray', zorder=-1)
    # ax00.plot([bls[K_idx], 2000],
    #           [0, 0], ls='--', lw=0.5,
    #           color='gray', zorder=-1)
    ax00.set_xlim(ax00x, ax00y)
    ax1.set_xlim(ax00x, ax00y)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_yticklabels(['0', '0.5', '1'])
    ax0.set_ylabel('ROS level (a.u.)')
    ax00.spines['top'].set_visible(False)
    ax1.set_ylim([-0.3, 0.3])
    ax1.set_yticks([-0.3, 0, 0.3])
    ax1.set_yticklabels(['-0.3', '0', '0.3'])
    ax1.set_ylabel('MS (a.u.)')
    lns = lm+ll
    labs = [ll.get_label() for ll in lns]
    ax0.legend(lns, labs, frameon=False, handlelength=0.5,
               loc='lower left', ncol=1, borderaxespad=0)
    for axx in [ax0, ax1]:
        axx.set_xlabel(r'Non-spiking costs (%s)' % kANT_units)
        axx.spines['top'].set_visible(False)
        axx.spines['right'].set_visible(False)
        fp.add_logticks(axx)
    return


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)


def clean_ax(axs):
    for ax in axs:
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


def plot_summary(gss, fet_cases, ret_cases):
    # gs = gss[:, 2]
    # gsx = gss[3, ]
    filename_prefix_ret = 'refrac_6_rise_5.0'
    filename_prefix_fet = 'iclamp2'
    data = np.load('./spike_compensation/spike_compensate_summary_' +
                   filename_prefix_ret + '.npz')
    mode = data['mode']
    spike_quanta = data['spike_quanta']
    mito_baseline = data['mito_baseline']
    mode_cmap = colors.ListedColormap(Set3_4_r.mpl_colors)
    ret_mode_dict = {0: 'Bursting', 3: 'Continuous',
                     2: 'Regular', 1: 'Silent'}
    mode_cmap = colors.ListedColormap([fp.colormap_dict[ret_mode_dict[cc]]
                                       for cc in range(4)])
    mode_bounds = [0, 1, 2, 3, 4]
    mode_norm = colors.BoundaryNorm(mode_bounds, mode_cmap.N)
    ax1 = plt.subplot(gs[2, 0])
    ax1.imshow(mode, origin='lower', aspect='auto',
               cmap=mode_cmap, norm=mode_norm)
    for ii, case in enumerate(ret_cases):
        if ii > 0:
            xx = np.searchsorted(mito_baseline, case.bl, side='left')
            yy = np.searchsorted(spike_quanta, case.q, side='left')
            ax1.plot(xx-0.5, yy-0.5, marker='*', clip_on=False,
                     color=fp.ln_cols_ret[ii], markersize=7,
                     markeredgecolor='k', markeredgewidth=0.3,  zorder=10)

    fix_axis_ticks([ax1], mito_baseline, spike_quanta)
    # fet cases
    data = np.load('./spike_compensation/spike_compensate_summary_fet_' +
                   filename_prefix_fet + '.npz')
    mode = data['mode']
    spike_quanta = data['spike_quanta']
    mito_baseline = data['mito_baseline']
    fet_mode_dict = {2: 'Regular', 3: 'Silent', 4: 'Adapting'}
    mode_cmap = colors.ListedColormap([fp.colormap_dict[fet_mode_dict[cc]]
                                       for cc in [2, 3, 4]])
    mode_bounds = [2, 3, 4, 5]
    mode_norm = colors.BoundaryNorm(mode_bounds, mode_cmap.N)
    ax2 = plt.subplot(gs[2, 1])
    ax2.imshow(mode, origin='lower', aspect='auto',
               cmap=mode_cmap, norm=mode_norm)
    for ii, case in enumerate(fet_cases):
        if ii < 2:
            xx = np.searchsorted(mito_baseline, case.bl, side='left')
            yy = np.searchsorted(spike_quanta, case.q, side='left')
            ax2.plot(xx-0.5, yy-0.5, marker='*', clip_on=False,
                     color=fp.ln_cols_fet[ii], markersize=7,
                     markeredgecolor='k', markeredgewidth=0.3,  zorder=10)
    cax2 = plt.subplot(gs[3, :])
    # cax2 = plt.subplot(gs[1, :2])
    cbar_labels = ['Continuous', 'Bursting',
                   'Regular', 'Silent',
                   'Adapting']
    mode_cmap = colors.ListedColormap([fp.colormap_dict[cc]
                                       for cc in cbar_labels])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, mode_cmap.N)
    cbar = colorbar.ColorbarBase(cax2, cmap=mode_cmap,
                                 norm=norm,
                                 spacing='proportional',
                                 orientation='horizontal')
    for j, lab in enumerate(cbar_labels):
        # if j == 0:
        #     cbar.ax.text(j+1, -8, lab, ha='right',
        #                  va='center', rotation=0, clip_on=False)
        # else:
        cbar.ax.text(j+0.5, -8, lab, ha='center',
                     va='center', rotation=0, clip_on=False)
    cbar.ax.set_xticklabels([])
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)
    fix_axis_ticks([ax2], mito_baseline, spike_quanta)
    # ax2.set_yticklabels([])
    ax1.set_ylabel('Per-spike cost (a.u.)')
    ax2.set_ylabel('Per-spike cost (a.u.)')
    ax1.set_xlabel('Non-spiking costs (%s)' % kANT_units,
                   fontsize=7, color='k')
    ax2.set_xlabel('Non-spiking costs (%s)' % kANT_units,
                   fontsize=7, color='k')

    # ax1.text(0.6, -0.25, 'Non-spiking costs (%s)' % kANT_units,
    #          fontsize=7, color='k', transform=ax1.transAxes).set_clip_on(False)
    return


def fix_axis_ticks(axs, mito_baseline, spike_quanta):
    for ax in axs:
        ax.set_xticks(np.arange(0, len(mito_baseline), 4))
        labels = ["{:d}".format(int(x)) for ii, x in enumerate(mito_baseline)
                  if ii % 4 == 0]
        ax.set_xticklabels(labels)
    axs[0].set_yticks(np.arange(len(spike_quanta))[::2])
    labels = ["{:.1f}".format(x*100) for x in spike_quanta]
    axs[0].set_yticklabels(labels[::2])
    for ax in axs[1:]:
        ax.set_yticks(np.arange(len(spike_quanta))[::2])
        labels = ["" for x in spike_quanta]
        ax.set_yticklabels(labels[::2])
    fontprops = fm.FontProperties(size=6)
    axs[0].text(0., 1.05, s="1e-2", ha='left', va='center',
                transform=axs[0].transAxes, fontproperties=fontprops)

    
def single_run(case):
    figsize = fp.cm_to_inches([8.7, 6])
    fig = plt.figure(figsize=figsize)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(5, 1, wspace=0.2, hspace=0.2,
                           height_ratios=[0.5, 0.75, 1, 1, 1])
    if case.test_type == 'ret':
        spikes, t, r_mito, ros_vals, ms_vals = ret_run_sim(case,
                                                           theta_ret=0.025)
    elif case.test_type == 'fet':
        spikes, t, r_mito, ros_vals, ms_vals, i_stim = fet_run_sim(case,
                                                                   theta_fet=-0.05)
    ax1 = plt.subplot(gs[1, 0])
    ax1.plot(spikes, [0]*len(spikes), 'ko', markersize=1)
    ax1.set_xlim([0, 2500])
    ax2 = plt.subplot(gs[2, 0])
    ax2.plot(t, ros_vals)
    ax3 = plt.subplot(gs[3, 0])
    ax3.plot(t, r_mito.out['atp'])
    ax4 = plt.subplot(gs[4, 0])
    ax4.plot(t, ms_vals)
    print(np.mean(ros_vals[20000:]))
    plt.show()


class TestBLCases(object):
    def __init__(self, bl, q, psi_fac=1e-3, refrac=6,
                 align_spike_at=None,
                 title_text='', test_type='fet'):
        self.bl = bl
        self.q = q
        self.psi_fac = psi_fac
        self.refrac = refrac
        if not align_spike_at:
            self.start = False
            self.end = False
        else:
            if test_type == 'fet':
                self.start = align_spike_at - 100
                self.end = self.start + 700
            else:
                self.start = align_spike_at - 50
                self.end = self.start + 300
        self.title_text = title_text
        self.test_type = test_type

        
if __name__ == '__main__':
    kANT_units = '10$^{-3}$/s'
    figsize = fp.cm_to_inches([8.9, 17])
    fig = plt.figure(figsize=figsize)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(4, 2, hspace=1,
                           height_ratios=[2, 5, 2.5, 0.1])
    # gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=1,
    #                                        width_ratios=[1, 1],
    #                                        subplot_spec=gs[0, :])
    # MS map
    metabolic_signal_plot(gs, theta_ret=0.025, theta_fet=-0.05)

    # # # # # ret cases
    gs1 = gridspec.GridSpecFromSubplotSpec(4, 1,
                                           height_ratios=[1.25, 1, 1, 1],
                                           subplot_spec=gs[1, 0])

    at_min = TestBLCases(bl=30, q=0.06, test_type='ret',
                         align_spike_at=427.31,
                         title_text='30,0.06')
    at_min_bl = TestBLCases(bl=30, q=0.012, test_type='ret',
                            align_spike_at=542.46,
                            title_text='30,0.012')
    at_min_bl_fast = TestBLCases(bl=30, q=0.012, test_type='ret',
                                 refrac=2,
                                 align_spike_at=587.17,
                                 title_text='30,0.012')
    ret_cases = [at_min_bl_fast, at_min_bl, at_min]
    ret_cases_considered(gs1, ret_cases)
    

    # # # # fet cases
    gs2 = gridspec.GridSpecFromSubplotSpec(4, 1,
                                           height_ratios=[1.25, 1, 1, 1],
                                           subplot_spec=gs[1, 1])
    at_min = TestBLCases(bl=80, q=0.06, test_type='fet',
                         align_spike_at=700.0,
                         title_text='80,0.06')
    at_max_q = TestBLCases(bl=80, q=0.15, test_type='fet',
                           align_spike_at=700.0,
                           title_text='80,0.15')
    fet_cases = [at_max_q, at_min]
    fet_cases_considered(gs2, fet_cases, theta_fet=-0.05)

    plot_summary([gs], fet_cases, ret_cases)

    gs.tight_layout(fig)
    plt.savefig('Figure2n.png', dpi=300, transparent=False)
    #plt.show()
