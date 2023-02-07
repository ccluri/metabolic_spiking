import numpy as np
from brian2 import ms, second, Hz, mV
import pickle

import avalan_props as avaln

import figure_properties as fp
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib
matplotlib.use('Agg')


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def summary_spikes(total_neurons, sim_time, times, nidx, M, dict_key,
                   dict_entry, filename='', opt_dicts=[], curr_filename=''):
    seed, we, wi, Amax, Atau, K_mu, K_sig = dict_key.split('_')
    start_time = 0.475*sim_time/ms
    end_time = (0.1+0.475)*sim_time/ms
    figsize = fp.cm_to_inches([8.9, 19])
    fig = plt.figure(figsize=figsize)  # width x height
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(7, 4, wspace=0.4, hspace=0.,
                           width_ratios=[1, 1, 1, 1],
                           height_ratios=[1, 0.5, 0.8, 0.5, 0.5, 1, 1])  # row x column
    gsx = gridspec.GridSpecFromSubplotSpec(4, 1, hspace=0.4,
                                           height_ratios=[0.5, 0.8, 0.5, 0.5],
                                           subplot_spec=gs[1:5, :])
    gsy = gridspec.GridSpecFromSubplotSpec(1, 2,
                                           width_ratios=[0.7, 0.3],
                                           subplot_spec=gs[0, :])
    ax1 = plt.subplot(gsy[0, 1])  # Default status; M plot
    ax0 = plt.subplot(gsx[0, :])  # population activity summary
    ax2 = plt.subplot(gsx[1, :])  # Colored raster (M vals)
    ax3 = plt.subplot(gsx[2, :], sharex=ax2)  # Current values of the zoom
    ax3c = plt.subplot(gsx[3, :])  # Current values of the zoom
    # ax3c.plot([0, -100], [0, -100])
    Amax = int(Amax)
    Minf = lambda frac : (2 / (1 + np.exp(-8*(frac-1.)))) - 1
    test_ks = np.geomspace(0.1, 10, num=100)
    ax1.plot(test_ks, Minf((2/(1+test_ks))),
             c='#d95f02', lw=0.5)
    ax1.set_ylabel('I$_{metabolic}$', labelpad=-7)
    ax1.text(-0.2, 1.1, s='(nA)', color='k', va='center', ha='center',
             transform=ax1.transAxes, clip_on=False)
    # set_ylabel('%', loc='top', rotation=0, labelpad=-5)
    ax1.set_xscale('log')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels([-Amax/100, 0, Amax/100])
    ax1.set_xticks([0.1, 1, 10])
    ax1.set_ylim([-1.2, 1.2])
    ax1.plot(0.5, -1.2, marker='*', c='k', clip_on=False, markersize=7,
             markeredgecolor='none')
    ax1.plot(2, -1.2, marker='*', clip_on=False, color='gold', markersize=7,
             markeredgecolor='k', markeredgewidth=0.5, zorder=10)
    ax1.set_xlabel('x'+r'<ATP$_{syn}$>')
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 4, wspace=0.4, hspace=0.3,
                                           width_ratios=[1, 1, 1, 1],
                                           subplot_spec=gs[5, :])
    # gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, wspace=0.4, hspace=0.3,
    #                                        width_ratios=[1, 1, 1, 1],
    #                                        subplot_spec=gs[6, :])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, wspace=0.4, hspace=0,
                                           width_ratios=[1, 1, 1],
                                           subplot_spec=gs[6, :])
    ax4 = plt.subplot(gs0[0, 0])  # Average fr
    ax5 = plt.subplot(gs0[0, 1])  # ISI
    ax6 = plt.subplot(gs0[0, 2])  # CV
    ax7 = plt.subplot(gs0[0, 3])  # M

    ax8 = plt.subplot(gs1[0, 0])  # Avalanches time
    ax9 = plt.subplot(gs1[0, 1])  # Avalanches pop
    ax91 = plt.subplot(gs1[0, 2])  # Avalanches space
    # ax10 = plt.subplot(gs1[0, 2])  # ISI vs FR
    # ax11 = plt.subplot(gs1[0, 3])  # ISI_n vs ISI_n+1
    times_zoom = times[np.logical_and((times >= start_time),
                                      (times < end_time))]
    nidx_zoom = nidx[np.logical_and((times >= start_time), (times < end_time))]
    M_zoom = M[np.logical_and((times >= start_time), (times < end_time))]
    cmap_ = colors.ListedColormap(['#d01c8b', '#f1b6da', '#f7f7f7',
                                   '#b8e186', '#4dac26'])
    bounds = np.linspace(-0.5, 0.5, 6)
    norm_1 = colors.BoundaryNorm(bounds, cmap_.N, clip=True)
    im = ax2.scatter(times_zoom, nidx_zoom, c=M_zoom, cmap=cmap_, s=1,
                     linewidth=0.1, edgecolor='k', norm=norm_1)
    # ax2.plot([0, sim_time/2/ms], [4050, 4050],
    #          lw=2, c='gold', zorder=10)
    ax2.plot([0, sim_time/2/ms], [4050, 4050],
             lw=2, c='#e6ab02', zorder=10)
    ax2.plot([sim_time/2/ms, sim_time/ms], [4050, 4050],
             lw=2, c='k', zorder=10)
    cax = inset_axes(ax2, width='100%', height='100%',
                     loc='center',
                     bbox_to_anchor=(0.25, -0.05, 0.5, 0.03),
                     bbox_transform=ax2.transAxes)
    cb = plt.colorbar(im, cax=cax, ticks=[-0.3, -0.1, 0.1, 0.3],
                      orientation='horizontal', extend='both')
    xy_inset = (start_time, 4000)
    wd_inset = end_time - start_time
    ht_inset = 50
    mask = Rectangle(xy_inset, wd_inset, ht_inset,
                     edgecolor='None', facecolor='white', lw=0.5, zorder=8)
    ax2.add_patch(mask)
    ax2.set_xlim([start_time, end_time])
    ax2.set_ylim([3000, 4050])
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_yticks([3000, 4000])
    ax2.set_yticklabels(['3K', '4K'])
    ymin, ymax = ax2.get_ybound()
    asb = AnchoredSizeBar(ax2.transData,
                          int(50),
                          '50 ms',
                          loc='lower left',
                          bbox_to_anchor=(0.9, -0.15),
                          bbox_transform=ax2.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax2.add_artist(asb)
    ax2.set_ylabel('Neuron index')
    
    train_isi = dict_entry['train_isi']
    ax0.plot(dict_entry['bins'], dict_entry['valids'] / total_neurons,
             'gray', lw=0.1)
    ax0.set_ylim([-0.001, 0.025])
    ax0.set_yticks([0, 0.01, 0.02])
    ax0.set_yticklabels(['0', '10', '20'])
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.set_xlim([0, sim_time/ms])
    ax0.set_xticks([])
    ax0.set_xticklabels([])
    ymin, ymax = ax0.get_ybound()
    asb = AnchoredSizeBar(ax0.transData,
                          int(1000),
                          '1 second',
                          loc='lower left',
                          bbox_to_anchor=(0.85, -0.15),
                          bbox_transform=ax0.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax0.add_artist(asb)
    ax0.set_ylabel('Pop. act. (Hz)')
    # ax0.plot([0, sim_time/2/ms], [0.02, 0.02],
    #          lw=1.5, c='gold')
    ax0.plot([0, sim_time/2/ms], [0.02, 0.02],
             lw=1.5, c='#e6ab02')
    ax0.text(sim_time/4/ms, 0.03, s='Poisson input (3Hz)',
             color='k', va='center', ha='center')
    ax0.plot([sim_time/2/ms, sim_time/ms], [0.02, 0.02], lw=1.5, c='k')
    ax0.text(sim_time*0.75/ms, 0.03, s='No external input',
             color='k', va='center', ha='center')
    xy_inset = (start_time, 0)
    wd_inset = end_time - start_time
    ht_inset = (ymax)*0.6
    p = Rectangle(xy_inset, wd_inset, ht_inset,
                  edgecolor='k', facecolor='none', lw=0.5, zorder=10)
    ax0.add_patch(p)
    ccp = ConnectionPatch(xyA=xy_inset, xyB=(0, 1), axesA=ax0, axesB=ax2,
                          # connectionstyle="angle3,angleA=0,angleB=90",
                          coordsA="data", coordsB="axes fraction", lw=0.3)
    ax0.add_patch(ccp)
    ccp2 = ConnectionPatch(xyA=(xy_inset[0] + wd_inset, xy_inset[1]),
                           xyB=(1, 1), axesA=ax0, axesB=ax2,
                           # connectionstyle="angle3,angleA=0,angleB=90",
                           coordsA="data", coordsB="axes fraction", lw=0.3)
    ax0.add_patch(ccp2)

    with open(curr_filename, 'rb') as ff:
        data_curr = pickle.load(ff)
    test_ii = 0
    ax3.plot(data_curr['t']/ms, data_curr['Ie'][:, test_ii]/mV/100,
             c='#f04e4d', lw=0.5, label='excitatory')
    ax3.plot(data_curr['t']/ms, data_curr['Ii'][:, test_ii]/mV/100,
             c='#3465a4', lw=0.5, label='inhibitory')
    # 5 ms smoothing
    ax3.set_ylabel('Current (nA)')
    ax3.set_ylim([-4, 5])
    ax3.set_yticks([-3, 0, 3])
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.legend(frameon=False, ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, -0.4))
    ymin, ymax = ax3.get_ybound()
    asb = AnchoredSizeBar(ax3.transData,
                          int(50),
                          '50 ms',
                          loc='lower left',
                          bbox_to_anchor=(0.9, -0.2),
                          bbox_transform=ax3.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax3.add_artist(asb)

    ax3c.plot(data_curr['t']/ms, runningMeanFast((data_curr['Ie'][:, test_ii]/mV +
                                                  data_curr['Ii'][:, test_ii]/mV +
                                                  Amax*data_curr['M'][:, test_ii])/100, 50),
              c='gray', lw=0.5, label='net', zorder=9)
    ax3c.plot(data_curr['t']/ms, runningMeanFast(Amax*data_curr['M'][:, test_ii]/100, 50),
              c='#d95f02', lw=0.5, label='metabolic', zorder=10)
    ax3c.plot(data_curr['t']/ms, runningMeanFast((data_curr['Ie'][:, test_ii]/mV +
                                                  data_curr['Ii'][:, test_ii]/mV)/100, 50),
              c='#7D26CD', lw=0.5, label='synaptic', zorder=8)
    ax3c.set_ylabel(' Current (nA)')
    ax3c.set_ylim([-0.6, 0.6])
    # ax32.set_yticks([-0.6, 0, 0.6])
    ax3c.set_xlim([start_time, end_time])
    ax3c.set_xticks([])
    ax3c.set_xticklabels([])
    ax3c.spines['bottom'].set_visible(False)
    ax3c.spines['top'].set_visible(False)
    ax3c.spines['left'].set_visible(False)
    ax3c.legend(frameon=False, ncol=3, loc='lower center',
                bbox_to_anchor=(0.5, -0.4))
    ymin, ymax = ax3c.get_ybound()
    asb = AnchoredSizeBar(ax3c.transData,
                          int(50),
                          '50 ms',
                          loc='lower left',
                          bbox_to_anchor=(0.9, -0.2),
                          bbox_transform=ax3c.transAxes,
                          pad=0., borderpad=.0, sep=2,
                          frameon=False, label_top=False,
                          size_vertical=(ymax-ymin)/1000)
    ax3c.add_artist(asb)
    align_axis_labels([ax0, ax2, ax3, ax3c], axis='y', value=-0.07)
    
    avg_fr = dict_entry['avg_fr']
    train_isi = dict_entry['train_isi']
    cvs = dict_entry['cvs']
    if opt_dicts:
        avg_fr_fh = opt_dicts[0]['avg_fr']
        all_isi_fh = np.concatenate(opt_dicts[0]['train_isi'])
        cvs_fh = opt_dicts[0]['cvs']
        avg_fr_sh = opt_dicts[1]['avg_fr']
        all_isi_sh = np.concatenate(opt_dicts[1]['train_isi'])
        cvs_sh = opt_dicts[1]['cvs']

    N, B = np.histogram(np.array(avg_fr), bins=np.linspace(0, 40, 41))
    if opt_dicts:
        Nfh, B = np.histogram(np.array(avg_fr_fh), bins=np.linspace(0, 40, 41))
        Nsh, B = np.histogram(np.array(avg_fr_sh), bins=np.linspace(0, 40, 41))
        # ax4.plot(B[:-1], Nfh / np.sum(Nfh), c='gold', lw=.75)
        ax4.plot(B[:-1], Nfh / np.sum(Nfh), c='#e6ab02', lw=.75)
        ax4.plot(B[:-1], Nsh / np.sum(Nsh), c='k', lw=0.75)
        # ax4.plot((np.mean(avg_fr_fh)*Hz, np.mean(avg_fr_fh)*Hz),
        #          (0, np.max(N)/np.sum(N)), c='gold', lw=.75, ls='--')
        ax4.plot((np.mean(avg_fr_fh)*Hz, np.mean(avg_fr_fh)*Hz),
                 (0, np.max(N)/np.sum(N)), c='#e6ab02', lw=.75, ls='--')
        ax4.plot((np.mean(avg_fr_sh)*Hz, np.mean(avg_fr_sh)*Hz),
                 (0, np.max(N)/np.sum(N)), c='k', lw=.75, ls='--')
    ax4.set_xlabel('Avg firing rate(Hz)')
    ax4.set_xticks([0, 5, 10, 15, 20])
    ax4.set_yticks([0, 0.1, 0.2])
    ax4.set_yticklabels([0, 10, 20])
    ax4.set_ylabel('%', loc='top', rotation=0, labelpad=-5)
    ax4.set_xticklabels([0, '', 10, '', 20])
    ax4.set_xlim([-5, 27])
    ax4.set_ylim([-0.01, 0.22])
    all_isi = np.concatenate(train_isi)
    N, B = np.histogram(all_isi, bins=np.logspace(0, 3.1, 30))
    if opt_dicts:
        Nfh, B = np.histogram(all_isi_fh, bins=np.logspace(0, 3.1, 30))
        Nsh, B = np.histogram(all_isi_sh, bins=np.logspace(0, 3.1, 30))
        # ax5.plot(B[:-1], Nfh / np.sum(Nfh), c='gold', lw=.75)
        ax5.plot(B[:-1], Nfh / np.sum(Nfh), c='#e6ab02', lw=.75)
        ax5.plot(B[:-1], Nsh / np.sum(Nsh), c='k', lw=.75)
        # ax5.plot((np.mean(all_isi_fh), np.mean(all_isi_fh)),
        #          (0, np.max(N)/np.sum(N)), c='gold', lw=.75, ls='--')
        ax5.plot((np.mean(all_isi_fh), np.mean(all_isi_fh)),
                 (0, np.max(N)/np.sum(N)), c='#e6ab02', lw=.75, ls='--')
        ax5.plot((np.mean(all_isi_sh), np.mean(all_isi_sh)),
                 (0, np.max(N)/np.sum(N)), c='k', lw=.75, ls='--')
    ax5.set_xscale('symlog')
    ax5.set_xlabel('ISI (ms)')
    ax5.set_yticks([0, 0.05, 0.1])
    ax5.set_yticklabels([0, 5, 10])
    ax5.set_ylabel('%', loc='top', rotation=0, labelpad=-5)

    N, B = np.histogram(cvs, bins=np.linspace(0, 3, 30))
    if opt_dicts:
        Nfh, B = np.histogram(cvs_fh, bins=np.linspace(0, 3, 30))
        Nsh, B = np.histogram(cvs_sh, bins=np.linspace(0, 3, 30))
        # ax6.plot(B[:-1], Nfh / np.sum(Nfh), c='gold', lw=0.75)
        ax6.plot(B[:-1], Nfh / np.sum(Nfh), c='#e6ab02', lw=0.75)
        ax6.plot(B[:-1], Nsh / np.sum(Nsh), c='k', lw=0.75)
        # ax6.plot((np.mean(cvs_fh), np.mean(cvs_fh)),
        #          (0, np.max(N)/np.sum(N)), c='gold',
        #          lw=0.75, ls='--')
        ax6.plot((np.mean(cvs_fh), np.mean(cvs_fh)),
                 (0, np.max(N)/np.sum(N)), c='#e6ab02',
                 lw=0.75, ls='--')
        ax6.plot((np.mean(cvs_sh), np.mean(cvs_sh)),
                 (0, np.max(N)/np.sum(N)), c='k',
                 lw=0.75, ls='--')

    ax6.set_xlabel('CV ISI')
    ax6.set_xticks([0, 1, 2, 3])
    ax6.set_yticks([0, 0.1, 0.2])
    ax6.set_yticklabels([0, 10, 20])
    ax6.set_ylabel('%', loc='top', rotation=0, labelpad=-5)

    N, B = np.histogram(M, bins=np.linspace(-2, 1, 31))
    if opt_dicts:
        Mfh, B = np.histogram(opt_dicts[2], bins=np.linspace(-2, 1, 31))
        Msh, B = np.histogram(opt_dicts[3], bins=np.linspace(-2, 1, 31))
        # ax7.plot(B[:-1], Mfh / sum(Mfh), c='gold', lw=0.75)
        ax7.plot(B[:-1], Mfh / sum(Mfh), c='#e6ab02', lw=0.75)
        ax7.plot(B[:-1], Msh / sum(Msh), c='k', lw=0.75)
        # ax7.plot((np.mean(opt_dicts[2]),
        #           np.mean(opt_dicts[2])), (0, np.max(N)/np.sum(N)),
        #          c='gold', lw=0.75, ls='--')
        ax7.plot((np.mean(opt_dicts[2]),
                  np.mean(opt_dicts[2])), (0, np.max(N)/np.sum(N)),
                 c='#e6ab02', lw=0.75, ls='--')
        ax7.plot((np.mean(opt_dicts[3]),
                  np.mean(opt_dicts[3])), (0, np.max(N)/np.sum(N)),
                 c='k', lw=0.75, ls='--')
    ax7.set_xlabel('MS')
    ax7.get_xaxis().set_label_coords(0.5, -0.25)
    ax7.set_xticks([-2, -1, 0, 1])
    ax7.set_xlim([-2, 1])
    ax7.set_yticks([0, 0.05, 0.1, 0.15])
    ax7.set_yticklabels([0, 5, 10, 15])
    ax7.set_ylabel('%', loc='top', rotation=0, labelpad=-5)

    fits_t_sh = opt_dicts[1]['fit_t']
    fits_t_sh.plot_pdf(ax=ax8, color='k', markersize=1, marker='o',
                       linestyle='None')
    fits_t_sh.power_law.plot_pdf(ax=ax8, lw=0.5, color='k')
    fits_t_fh = opt_dicts[0]['fit_t']
    # fits_t_fh.plot_pdf(ax=ax8, color='gold', markersize=1,
    #                    marker='o')
    fits_t_fh.plot_pdf(ax=ax8, color='#e6ab02', markersize=1,
                       marker='o')
    ax8.text(0.6, 0.95, s=r'$\alpha_{D}=$'+str(opt_dicts[1]['talpha'])[:4],
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax8.transAxes, color='k')
    ax8.set_xlabel('Avalanche dur.\n(ms)')
    ax8.set_ylabel('P(D)', loc='top', rotation=0, labelpad=-12)
    ax8.set_xticks([1, 10, 100, 1000])
    ax8.set_ylim([2*10**-5, 2*10**0])
    ax8.set_yticks([10**-4, 10**-2, 10**0])

    fits_sh = opt_dicts[1]['fit_s']
    fits_sh.plot_pdf(ax=ax9, color='k', markersize=1, marker='o',
                     linestyle='None')
    fits_sh.power_law.plot_pdf(ax=ax9, lw=0.5, color='k')
    fits_fh = opt_dicts[0]['fit_s']
    # fits_fh.plot_pdf(ax=ax9, color='gold', markersize=1,
    #                  marker='o')
    fits_fh.plot_pdf(ax=ax9, color='#e6ab02', markersize=1,
                     marker='o')
    ax9.text(0.6, 0.95, s=r'$\alpha_{S}=$'+str(opt_dicts[1]['palpha'])[:4],
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax9.transAxes, color='k')
    ax9.set_xlabel('Avalanche size')
    ax9.set_ylabel('P(S)', loc='top', rotation=0, labelpad=-12)
    ax9.set_xticks([1, 100, 10000, 1000000])
    ax9.set_ylim([2*10**-7, 2*10**0])

    # # Plot ISI vs FR, ISI(n) vs ISI(n+1)
    # frate_sh = opt_dicts[1]['avg_fr']
    # train_isi_sh = opt_dicts[1]['train_isi']
    # inds_sh = np.argsort(frate_sh)

    # all_spikes_x = []
    # all_spikes_y = []
    # th = []
    # ys = []
    # for ii in range(10000):
    #     tt = train_isi[inds_sh[9999-ii]]
    #     th.append(tt)
    #     ys.append([ii]*len(tt))
    #     all_spikes_x.append(train_isi_sh[ii][:-1])
    #     all_spikes_y.append(train_isi_sh[ii][1:])
    # th = np.hstack(th)
    # ys = np.hstack(ys)
    # all_spikes_x = np.hstack(all_spikes_x)
    # all_spikes_y = np.hstack(all_spikes_y)
    # ax10.scatter(th, ys, c='k', marker=',',
    #              alpha=0.025, s=0.08)
    # ax10.set_xscale('log')
    # ax10.set_xlabel('ISI (ms)')
    # ax10.set_ylabel('Neuron idx')
    # ax11.scatter(all_spikes_x, all_spikes_y, c='k',
    #              alpha=0.025, s=0.08, marker=',')
    # ax11.set_xscale('log')
    # ax11.set_yscale('log')
    # ax11.set_xlabel(r'ISI$_n$ (ms)')
    # ax11.set_ylabel(r'ISI$_(n+1)$ (ms)')

    
    avaln_s = opt_dicts[1]['avalns']
    avaln_t = opt_dicts[1]['avalns_t']
    
    # logx = np.log10(avaln_t)
    # logy = np.log10(avaln_s)
    # coeffs = np.polyfit(logx, logy, deg=3)
    # poly = np.poly1d(coeffs)
    # yfit = lambda x: 10**(poly(np.log(x)))
    # ax91.loglog(avaln_t, yfit(avaln_t), lw=0.5, color='k')
    
    # slope = opt_dicts[1]['beta_fit'][0]
    # intercept = opt_dicts[1]['beta_fit'][1]
    # test_pts = [1, 1000]
    # op_pts = [(slope*ii)+intercept for ii in test_pts]
    # ax91.loglog(test_pts, op_pts, lw=0.5, color='k')
    # ax91.text(0.1, 0.95, s=r'1/$\gamma_{fit}=$'+str(opt_dicts[1]['beta_fit'][0])[:4],
    #           horizontalalignment='left',
    #           verticalalignment='center',
    #           transform=ax91.transAxes, color='k')

    ax91.loglog(avaln_t, avaln_s, color='k', markersize=1, marker='o',
                linestyle='None')
    ax91.set_ylabel('Avalanche size')
    ax91.set_xlabel('Avalanche dur.\n(ms)')
    ax91.set_xticks([1, 10, 100, 1000])
    ax91.set_yticks([1, 100, 10000, 1000000])

    # ax9.set_xticks([1, 100, 10000, 1000000])
    # ax9.set_ylim([2*10**-7, 2*10**0])
    # ax9.set_yticks([10**-4, 10**-2, 10**0])
    # ax3c.plot([0, 100], [0, 100])

    gs.tight_layout(fig)
    align_axis_labels([ax4, ax5, ax6, ax7], axis='x', value=-0.3)
    # align_axis_labels([ax8, ax9, ax91], axis='x', value=-0.5)
    neat_axs([ax0, ax1, ax2, ax3, ax3c,
              ax4, ax5, ax6, ax7, ax8, ax9, ax91])
    # neat_axs([ax0, ax1, ax2, ax3, ax3c,
    #           ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11])  # ax91])

    if len(filename) > 0:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)


def neat_axs(ax_list):
    for ax in ax_list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def bin_them(si_time, s_mon, bin_size):
    print('binning')
    times = s_mon['t']*1000 / second  # now in ms and unitless
    nidx = s_mon['i']
    si_time = si_time * 1000 / second  # ms but unitless
    # bin_size = 1.  # ms but unitless
    bins = np.arange(0, si_time, bin_size)
    valids = np.zeros_like(bins, dtype=int)
    for ii, jj in enumerate(bins):
        current_bin = nidx[((times > jj) & (times <= jj+bin_size))]
        valids[ii] = len(current_bin)
    return valids, bins


if __name__ == '__main__':
    total_neurons = 10000
    sim_time = 10*second
    bin_size = 1  # ms
    connectivity = 20
    dict_key = '20_0.3_5_25_300_200_50'
    filename = './netsim_results/' + str(connectivity) + '/nw_' + dict_key + '_poi_onoff_spks.pkl'
    curr_filename = './netsim_results/' + str(connectivity) + '/nw_' + dict_key + '_poi_onoff_currs.pkl'
    with open(filename, 'rb') as ff:
        times, nidx, M = avaln.load_sim_file(ff)
        # duration is unitless but in seconds
        dict_entry = avaln.process_each_file(times, nidx, M, duration=10)
        dict_fh, dict_sh, M_fh, M_sh = avaln.props_split(times, nidx,
                                                         M, sim_time)
        plot_filename = filename.rstrip('_spks.pkl') + '.png'
        plot_filename = 'Figure3_nw.png'
        summary_spikes(total_neurons, sim_time, times,
                       nidx, M, dict_key, dict_entry, filename=plot_filename,
                       opt_dicts=[dict_fh, dict_sh,
                                  M_fh, M_sh],
                       curr_filename=curr_filename)
