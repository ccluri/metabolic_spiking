import numpy as np

from utils import Recorder, Q_nak
from lifcell import LIFCell

import figure_properties as fp
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, patches, colors, colorbar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


fontprops = fm.FontProperties(size=6)


def double_spike(ax0):
    '''Dummy spikes'''
    params = {'deltaga': 0, 'refrc': 6,
              'Q': 1, 'init_atp': 1, 'g_nap': .25}
    c = LIFCell('test', **params)
    dt = 0.01
    time = 500
    t = np.arange(0, time, dt)
    i_inj = np.zeros_like(t)
    t_start = 150
    t_end = 155
    i_inj[int(t_start/dt):int(t_end/dt)] = 150
    t_start = 170
    t_end = 175
    i_inj[int(t_start/dt):int(t_end/dt)] = 150
    r = Recorder(c, ['v'], time, dt)
    for i in range(len(t)):
        c.i_inj = i_inj[i]
        c.update_vals(dt)
        r.update(i)
    ax0.plot(t, r.out['v'], c='k', lw=0.5)
    ax0.set_ylabel('Memb. Pot.')
    ax0.annotate('',
                 xy=(150, 0), xycoords='data', color='k',
                 xytext=(110, 0), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", mutation_scale=10, lw=0.5,
                                 color='k'))
    ax0.annotate('$t_{ref}$ = ISI$_{min}$', va='center',
                 xy=(170, 0), xycoords='data', color='k',
                 xytext=(220, 0), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", mutation_scale=10, lw=0.5,
                                 color='k'))
    ymin, ymax = ax0.get_ybound()
    asb0 = AnchoredSizeBar(ax0.transData,
                           int(25),
                           '25 ms', bbox_to_anchor=(0.8, 0.1),
                           bbox_transform=ax0.transAxes,
                           loc='upper left', fontproperties=fontprops,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=True,
                           size_vertical=(ymax-ymin)/1000)
    ax0.add_artist(asb0)
    ax0.set_xlim(100, 300)
    ax0.set_ylim(-100, 30)
    ax0.plot(150, -90, marker='o', c='g', clip_on=False, markersize=5)
    ax0.plot(170, -90, marker='o', c='g', clip_on=False, markersize=5)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.get_xaxis().set_visible(False)
    return ax0


def quantum(ax1):
    '''Illustrating baseline plus Q atp->adp'''
    dt = 0.01
    tt = np.arange(0, 500, dt)
    Qval = np.zeros_like(tt)
    vals = Q_nak(tt, 30)
    Qval[int(150/dt):] += vals[:len(Qval[int(150/dt):])]
    Qval[int(175/dt):] += vals[:len(Qval[int(175/dt):])]
    ax1.plot(tt, Qval, c='k', lw=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylim(-20, 70)
    ax1.set_ylabel(r'$ATP_C \rightarrow ADP_C$'+'\n(%s)' % Kant_units)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.plot(100, 0, marker='*', c='k', clip_on=False, markersize=7,
             markeredgecolor='none')
    ax1.text(s='+Q', x=120, y=27.5, fontsize=7)
    ymin, ymax = ax1.get_ybound()
    asb0 = AnchoredSizeBar(ax1.transData,
                           int(25),
                           '25 ms', bbox_to_anchor=(0.8, 0.2),
                           bbox_transform=ax1.transAxes,
                           loc='upper left', fontproperties=fontprops,
                           pad=0., borderpad=.0, sep=2,
                           frameon=False, label_top=True,
                           size_vertical=(ymax-ymin)/1000)
    ax1.add_artist(asb0)
    ax1.set_xlim(100, 300)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    

def spike_rise(ax, xlim, ylim):
    dt = 0.01
    tt = np.arange(0, 500, dt)
    t_peak = []
    for cc, rise_tm in zip(fp.ln_cols_rise, [0.1, 0.6, 1.2]):
        Qval = np.zeros_like(tt)
        vals = Q_nak(tt, 30, tau_rise=rise_tm)
        Qval[int(150/dt):] += vals[:len(Qval[int(150/dt):])]
        t_peak_val = tt[np.argmax(Qval)]-150
        t_peak.append(t_peak_val)
        ax.plot(tt, Qval, c=cc, label=str(int(t_peak_val)) + ' ms', lw=0.5)
    print(t_peak)
    ax.plot(150, -5, marker='o', c='g', clip_on=False, markersize=5)
    ax.plot([150, 150], [-5, 70], ls=':', c='g', lw=0.5)
    ax.annotate('$t_{lag}$', va='center',
                xy=(149.6, 35), xycoords='data', color=fp.ln_cols_rise[0],
                xytext=(149.6+t_peak[0]+1, 35), textcoords='data',
                arrowprops=dict(arrowstyle="|-|", mutation_scale=3, lw=1,
                                color=fp.ln_cols_rise[0]))
    ax.legend(frameon=False, loc='lower left', handlelength=0.5, ncol=1,
              title='$t_{lag}$', bbox_to_anchor=(0.6, -0.05),
              bbox_transform=ax.transAxes)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_summary_ret(gs, filename_prefix_ret='refrac_6_rise_0.6'):
    data = np.load('./spike_compensation/spike_compensate_summary_' + filename_prefix_ret + '.npz')
    cv = data['cv']
    netros = data['netros']
    fr = data['fr']
    spike_quanta = data['spike_quanta']
    mito_baseline = data['mito_baseline']
    ax1 = plt.subplot(gs[0, 0])
    rr = ax1.imshow(netros, origin='lower', cmap='Reds', vmin=0, vmax=1)
    # cax1 = plt.subplot(gs[1, 0])
    # cbar = plt.colorbar(rr, cax=cax1, orientation="horizontal")
    # cbar.set_ticks([0, 1])
    ax1.set_title('Average ROS (a.u.)', pad=5)
    ax2 = plt.subplot(gs[0, 1])
    fr_norm = colors.LogNorm(vmin=1, vmax=300)
    ii = ax2.imshow(fr*1000, origin='lower',
                    cmap=cm.viridis, norm=fr_norm)
    # cax2 = plt.subplot(gs[1, 1])
    # cbar = plt.colorbar(ii, cax=cax2, orientation="horizontal",
    #                     extend='min')
    # cbar.set_ticks([1, 10, 100])
    ax2.set_title('Firing rate (Hz)', pad=5)

    ax3 = plt.subplot(gs[0, 2])
    cmap = cm.RdYlBu
    new_cmap = fp.truncate_colormap(cmap, 0.15, 0.85)
    # cv_norm = colors.LogNorm(vmin=1, vmax=100)
    bb = ax3.imshow(cv, origin='lower', cmap=new_cmap, vmin=0, vmax=2)
    # cax3 = plt.subplot(gs[1, 2])
    # cbar = plt.colorbar(bb, cax=cax3, orientation="horizontal",
    #                     extend='max')
    # cbar.set_ticks([0, 1, 2])
    ax3.set_title('CV', pad=5)
    ax1.set_ylabel('Per-spike cost Q (%s)' % Kant_units)
    ax2.set_xlabel('Non-spiking costs (%s)' % Kant_units)
    fix_axis_ticks([ax1, ax2, ax3], mito_baseline, spike_quanta)
    return


def plot_summary_fet(gs, filename_prefix_fet):
    data = np.load('./spike_compensation/spike_compensate_summary_fet_' +
                   filename_prefix_fet + '.npz')
    data_bl = np.load('./spike_compensation/spike_compensate_summary_fet_ros_baseline.npz')
    netros_bl = data_bl['netros']
    cv = data['cv']
    netros = data['netros']
    fr = data['fr']
    spike_quanta = data['spike_quanta']
    mito_baseline = data['mito_baseline']
    ax1 = plt.subplot(gs[2, 0])
    rr = ax1.imshow(netros_bl-netros,
                    origin='lower', cmap='Reds', vmin=0, vmax=1)
    cax1 = plt.subplot(gs[1, 0])
    cbar = plt.colorbar(rr, cax=cax1, orientation="horizontal")
    cbar.set_ticks([0, 1])
    ax1.set_title(r'$\Delta$ ROS (a.u.)', pad=5)
    ax2 = plt.subplot(gs[2, 1])
    fr_norm = colors.LogNorm(vmin=1, vmax=300)
    ii = ax2.imshow(fr, origin='lower',
                    cmap=cm.viridis, norm=fr_norm)
    cax2 = plt.subplot(gs[1, 1])
    cbar = plt.colorbar(ii, cax=cax2, orientation="horizontal",
                        extend='min')
    cbar.set_ticks([1, 10, 100])
    ax2.set_title('Firing rate (Hz)', pad=5)
    ax3 = plt.subplot(gs[2, 2])
    cmap = cm.RdYlBu
    new_cmap = fp.truncate_colormap(cmap, 0.15, 0.85)
    bb = ax3.imshow(cv, origin='lower', cmap=new_cmap,
                    vmin=0, vmax=2)
    cax3 = plt.subplot(gs[1, 2])
    cbar = plt.colorbar(bb, cax=cax3, orientation="horizontal",
                        extend='max')
    cbar.set_ticks([0, 1, 2])
    ax3.set_title('CV', pad=5)
    ax1.set_ylabel('Per-spike cost Q (%s)' % Kant_units)
    ax2.set_xlabel('Non-spiking costs (%s)' % Kant_units)
    fix_axis_ticks([ax1, ax2, ax3], mito_baseline, spike_quanta)
    return


def plot_summary(ax1, filename_prefix_ret='refrac_6_rise_0.6'):
    # ret cases
    data = np.load('./spike_compensation/spike_compensate_summary_' + filename_prefix_ret + '.npz')
    mode = data['mode']
    spike_quanta = data['spike_quanta']
    mito_baseline = data['mito_baseline']
    ret_mode_dict = {0: 'Bursting', 3: 'Continuous',
                     2: 'Regular', 1: 'Silent'}
    mode_cmap = colors.ListedColormap([fp.colormap_dict[ret_mode_dict[cc]]
                                       for cc in range(4)])
    mode_bounds = [0, 1, 2, 3, 4]
    mode_norm = colors.BoundaryNorm(mode_bounds, mode_cmap.N)
    mm = ax1.imshow(mode, origin='lower', cmap=mode_cmap, norm=mode_norm)
    return mm, mito_baseline, spike_quanta


def fix_axis_ticks(axs, mito_baseline, spike_quanta,
                   xlabels=True, ylabels=True):
    for ax in axs:
        ax.set_xticks(np.arange(0, len(mito_baseline), 4))
        if xlabels:
            labels = ["{:d}".format(int(x)) for ii, x in enumerate(mito_baseline) if ii%4==0]
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels([])
    axs[0].set_yticks(np.arange(len(spike_quanta))[::2])

    labels = ["{0:.1f}".format(x) for x in spike_quanta]
    if ylabels:
        axs[0].set_yticklabels(labels[::2])
    else:
        axs[0].set_yticklabels([])
    for ax in axs[1:]:
        ax.set_yticks(np.arange(len(spike_quanta))[::2])
        labels = ["" for x in spike_quanta]
        ax.set_yticklabels(labels[::2])


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)


if __name__ == '__main__':
    Kant_units = '10$^{-3}$/s'
    figsize = fp.cm_to_inches([8.9, 21.5])
    fig = plt.figure(figsize=figsize)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(4, 3,
                           height_ratios=[3, 1.25, 3.07, 0.07])
    gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, wspace=0.2, hspace=0.5,
                                            height_ratios=[0.9, 0.07,
                                                           0.9],
                                            subplot_spec=gs[0, :])

    plot_summary_ret(gs00, filename_prefix_ret='refrac_6_rise_5.0')
    plot_summary_fet(gs00, filename_prefix_fet='iclamp2')

    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.2,
                                           width_ratios=[1, 1],
                                           subplot_spec=gs[1, :])
    ax_db_spk = plt.subplot(gs1[0, 0])
    double_spike(ax_db_spk)
    ax_atp_costs = plt.subplot(gs1[1, 0])
    quantum(ax_atp_costs)
    align_axis_labels([ax_atp_costs], axis='y', value=-0.1)
    xy_inset = (145, -5)
    wd_inset = 20
    ht_inset = 45
    ax_spike_rises = plt.subplot(gs1[:, 1])
    spike_rise(ax_spike_rises,
               xlim=[xy_inset[0], xy_inset[0]+wd_inset],
               ylim=[xy_inset[1], xy_inset[1]+ht_inset])
    rect1 = patches.Rectangle(xy_inset, wd_inset, ht_inset,
                              edgecolor='gray', linewidth=0.5,
                              facecolor='none')
    ax_atp_costs.add_patch(rect1)
    ccp = patches.ConnectionPatch(xyA=(xy_inset[0]+wd_inset, xy_inset[1]),
                                  xyB=(0, 0), axesA=ax_atp_costs,
                                  axesB=ax_spike_rises, lw=0.3, color='gray',
                                  coordsA="data", coordsB="axes fraction")
    ax_atp_costs.add_patch(ccp)
    ccp2 = patches.ConnectionPatch(xyA=(xy_inset[0]+wd_inset,
                                        xy_inset[1]+ht_inset),
                                   xyB=(0, 1),
                                   axesA=ax_atp_costs,
                                   axesB=ax_spike_rises,
                                   lw=0.3, color='gray',
                                   coordsA="data", coordsB="axes fraction")
    ax_atp_costs.add_patch(ccp2)
    title_texts = ['4 ms', '8 ms', '12 ms']
    axs = []
    gs2 = gridspec.GridSpecFromSubplotSpec(3, 3, wspace=0.2,
                                           width_ratios=[1, 1, 1],
                                           subplot_spec=gs[2, :])
    for ii, rise_tm in enumerate(['_rise_3.0', '_rise_5.0', '_rise_7.0']):
        for jj, refrac in enumerate([2, 6, 10]):
            filename_prefix = 'refrac_' + str(refrac) + rise_tm
            ax = plt.subplot(gs2[jj, ii])
            mm, mb, sq = plot_summary(ax, filename_prefix)
            xlabels = True if (jj == 2) else False
            ylabels = True if (ii == 0) else False
            fix_axis_ticks([ax], mb, sq, xlabels, ylabels)
            if jj == 0:
                ax.set_title('$t_{lag}$ = ' + title_texts[ii],
                             color=fp.ln_cols_rise[ii], pad=5)
            if ii == 0:
                if jj == 1:
                    ax.set_ylabel('Per-spike cost Q (%s)' % Kant_units)
            if ii == 2:
                ax.text(1.1, .5, '$t_{ref}$ = ' + str(refrac) + ' ms',
                        va='center', ha='left', clip_on=False,
                        transform=ax.transAxes, rotation=90)
            if jj == 2:
                if ii == 1:
                    ax.set_xlabel('Non-spiking costs (%s)' % Kant_units)
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, wspace=0.2,
                                           width_ratios=[.1, 1, 0.1],
                                           subplot_spec=gs[3, :])
    cax1 = plt.subplot(gs3[0, 1])
    cbar = plt.colorbar(mm, cax=cax1, orientation="horizontal")
    cbar_labels = ['Continuous', 'Bursting',
                   'Regular', 'Silent']
    mode_cmap = colors.ListedColormap([fp.colormap_dict[cc]
                                       for cc in cbar_labels])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, mode_cmap.N)
    cbar = colorbar.ColorbarBase(cax1, cmap=mode_cmap,
                                 norm=norm,
                                 spacing='proportional',
                                 orientation='horizontal')
    for j, lab in enumerate(cbar_labels):
        if j == 0:
            cbar.ax.text(j+1, -8, lab, ha='right',
                         va='center', rotation=0, clip_on=False)
        else:
            cbar.ax.text(j+0.5, -8, lab, ha='center',
                         va='center', rotation=0, clip_on=False)
    cbar.ax.set_xticklabels([])
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)
    gs.tight_layout(fig)
    plt.savefig('Figure2_supp.png', dpi=300)
    # plt.show()
