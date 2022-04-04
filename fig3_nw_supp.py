import glob
import pickle
import numpy as np

import figure_properties as fp
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors, cm


def fetch_parameter(path, seed, case):
    try:
        ff = open(path + '/' + str(seed) + '_' + case + '.npz')
        ff.close()
        print(path + '/' + str(seed) + '_' + case + '.npz')
        print('Found a summary file of this simulation, using that')
        pass
    except IOError:
        lst = glob.glob(path + str(seed) + '_*_*_'+case+'_summary.pkl')
        wii = np.arange(1, 11, 1)
        wee = np.arange(0.1, 1.1, 0.1)
        wii_l = len(wii)
        wee_l = len(wee)
        fr_avg_fh = np.zeros((wii_l, wee_l))
        fr_avg_sh = np.zeros((wii_l, wee_l))
        cvs_fh = np.zeros((wii_l, wee_l))
        cvs_sh = np.zeros((wii_l, wee_l))
        isi_fh = np.zeros((wii_l, wee_l))
        isi_sh = np.zeros((wii_l, wee_l))
        talpha_fh = np.zeros((wii_l, wee_l))
        talpha_sh = np.zeros((wii_l, wee_l))
        bf_fh = np.zeros((wii_l, wee_l))
        bf_sh = np.zeros((wii_l, wee_l))
        m_fh = np.zeros((wii_l, wee_l))
        m_sh = np.zeros((wii_l, wee_l))
        for ii, we in enumerate(wee):
            for jj, wi in enumerate(wii):
                for fname in lst:
                    fx = fname.split('/')[-1]
                    test_e, test_i = [float(ii) for ii in fx.split('_')[1:3]]
                    if np.isclose(test_i, wi) and np.isclose(test_e, we):
                        print(test_e, test_i, wi, we)
                        realf = fname
                with open(realf, 'rb') as ff:
                    fh, sh = pickle.load(ff)
                print(realf)
                fr_avg_fh[ii, jj] = np.mean(fh['avg_fr'])
                fr_avg_sh[ii, jj] = np.mean(sh['avg_fr'])
                cvs_fh[ii, jj] = np.mean(fh['cvs'])
                cvs_sh[ii, jj] = np.mean(sh['cvs'])
                isi_fh[ii, jj] = np.mean(np.concatenate(fh['train_isi']))
                isi_sh[ii, jj] = np.mean(np.concatenate(sh['train_isi']))
                m_fh[ii, jj] = fh['M_avg']
                m_sh[ii, jj] = sh['M_avg']
                talpha_fh[ii, jj] = fh['talpha']
                talpha_sh[ii, jj] = sh['talpha']
                bf_fh[ii, jj] = fh['branch_fac']
                bf_sh[ii, jj] = sh['branch_fac']
        np.savez(path + '/' + str(seed) + '_' + case + '.npz',
                 fr_avg_fh=fr_avg_fh, fr_avg_sh=fr_avg_sh,
                 cvs_fh=cvs_fh, cvs_sh=cvs_sh,
                 isi_fh=isi_fh, isi_sh=isi_sh,
                 m_fh=m_fh, m_sh=m_sh,
                 talpha_fh=talpha_fh, talpha_sh=talpha_sh,
                 bf_fh=bf_fh, bf_sh=bf_sh,
                 wii=wii, wee=wee)


def plot_colorbars(gs, fr, isis, cvs, m_avg, talpha):
    cax1 = plt.subplot(gs[1, 1:3])
    cbar = plt.colorbar(fr, cax=cax1, orientation='horizontal', extend='both')
    cbar.ax.set_title('Avg. firing rate (Hz)', pad=3)
    cax2 = plt.subplot(gs[3, 1:3])
    cbar = plt.colorbar(isis, cax=cax2, orientation='horizontal', extend='max')
    cbar.ax.set_title('Avg. ISI (ms)', pad=3)
    cax3 = plt.subplot(gs[5, 1:3])
    cbar = plt.colorbar(cvs, cax=cax3, orientation='horizontal', extend='max')
    cbar.ax.set_title('Avg. CV ISI', pad=3)
    cax4 = plt.subplot(gs[7, 1:3])
    cbar = plt.colorbar(m_avg, cax=cax4, orientation='horizontal',
                        extend='min')
    cbar.ax.set_title('Avg. MS', pad=3)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cax5 = plt.subplot(gs[9, 1:3])
    cbar = plt.colorbar(talpha, cax=cax5, orientation='horizontal',
                        extend='both')
    cbar.ax.set_title(r'$\alpha_D$ for P(D) distrb.', pad=3)
    cbar.set_ticks([1, 1.5, 2])
    return


def plot_summary(gs, path, seed, case):
    data = np.load(path+str(seed)+'_'+case+'.npz')
    ax1 = plt.subplot(gs[0, 0])
    cmap = cm.viridis
    fr_norm = colors.LogNorm(vmin=0.1, vmax=30)
    fr = ax1.imshow(data['fr_avg_fh'], origin='lower', cmap=cmap, norm=fr_norm)
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(data['fr_avg_sh'], origin='lower', cmap=cmap, norm=fr_norm)
    cmap = cm.cividis
    isi_norm = colors.LogNorm(vmin=1, vmax=300)
    ax3 = plt.subplot(gs[2, 0])
    isis = ax3.imshow(data['isi_fh'], origin='lower', cmap=cmap, norm=isi_norm)
    ax4 = plt.subplot(gs[2, 1])
    ax4.imshow(data['isi_sh'], origin='lower', cmap=cmap, norm=isi_norm)
    cmap = cm.RdYlBu
    new_cmap = truncate_colormap(cmap, 0.15, 0.85)
    ax5 = plt.subplot(gs[4, 0])
    cvs = ax5.imshow(data['cvs_fh'], origin='lower',
                     cmap=new_cmap, vmin=0, vmax=2)
    ax6 = plt.subplot(gs[4, 1])
    ax6.imshow(data['cvs_sh'], origin='lower', cmap=new_cmap, vmin=0, vmax=2)
    cmap = cm.PiYG
    new_cmap = truncate_colormap(cmap, 0.15, 0.85)
    ax7 = plt.subplot(gs[6, 0])
    m_avg = ax7.imshow(data['m_fh'], origin='lower',
                       cmap=new_cmap, vmin=-1, vmax=1)
    ax8 = plt.subplot(gs[6, 1])
    ax8.imshow(data['m_sh'], origin='lower', cmap=new_cmap, vmin=-1, vmax=1)
    cmap = cm.coolwarm
    ax9 = plt.subplot(gs[8, 0])
    talpha = ax9.imshow(data['talpha_fh'], origin='lower',
                        cmap=cmap, vmin=1, vmax=2)
    ax10 = plt.subplot(gs[8, 1])
    ax10.imshow(data['talpha_sh'], origin='lower', cmap=cmap, vmin=1, vmax=2)
    axs = [ax1, ax3, ax5, ax7, ax9]
    set_tickvals(axs, data['wee'], axis='y')
    axs = [ax2, ax4, ax6, ax8, ax10]
    remove_ticks(axs, 'y')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    set_tickvals(axs, data['wii'], axis='x')
    return axs, fr, isis, cvs, m_avg, talpha


def set_tickvals(axs, vals, axis='x'):
    for ax in axs:
        if axis == 'x':
            ax.set_xticks([0, 4, 8])
            ax.set_xticklabels([int(ii*10) for ii in
                                [vals[0], vals[4], vals[8]]])
        else:
            ax.set_yticks([0, 4, 8])
            ax.set_yticklabels([int(ii*10) for ii in
                                [vals[0], vals[4], vals[8]]])


def remove_ticks(axs, axis='x'):
    for ax in axs:
        if axis == 'x':
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

            
def draw_external_ip_line(axs, poi=True, text=True):
    for ax in axs:
        if poi:
            ax.plot([0, 1], [1.07, 1.07], lw=1.5, c='gold',
                    clip_on=False, transform=ax.transAxes)
            if text:
                ax.text(0.5, 1.2, s='Poisson input', transform=ax.transAxes,
                        color='k', va='center', ha='center', clip_on=False)
        else:
            ax.plot([0, 1], [1.07, 1.07], lw=1.5, c='k',
                    clip_on=False, transform=ax.transAxes)
            if text:
                ax.text(0.5, 1.2, s='No input', transform=ax.transAxes,
                        color='k', va='center', ha='center', clip_on=False)
    return axs


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


seed = 20
connectivity = 20
path = './netsim_results/' + str(connectivity) + '/'

figsize = fp.cm_to_inches([8.9, 21])
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(10, 4, wspace=0.7, hspace=0.2, height_ratios=[1, 0.07,
                                                                     1, 0.07,
                                                                     1, 0.07,
                                                                     1, 0.07,
                                                                     1, 0.07])

gsx = gridspec.GridSpecFromSubplotSpec(10, 2, subplot_spec=gs[:, :2],
                                       wspace=0.1,
                                       height_ratios=[1, 0.07,
                                                      1, 0.07,
                                                      1, 0.07,
                                                      1, 0.07,
                                                      1, 0.07])
case = '25_300_200_50'  # metabolic current based simulation
fetch_parameter(path, seed, case)
axs, fr, isis, cvs, m_avg, talpha = plot_summary(gsx, path, seed, case)
draw_external_ip_line(axs[0::2], poi=True, text=True)
draw_external_ip_line(axs[1::2], poi=False, text=True)

ax0 = axs[0]
for ax in axs[0::2]:
    ax.text(-0.3, 0.5, s='Excitatory (ns)',
            transform=ax.transAxes,
            va='center', ha='center', clip_on=False, rotation=90)
for ax in axs[0::2]:
    ax.text(1, -0.3, s='Inhibitory (ns)',
            transform=ax.transAxes,
            va='center', ha='center', clip_on=False)

gsy = gridspec.GridSpecFromSubplotSpec(10, 2, subplot_spec=gs[:, 2:],
                                       wspace=0.1,
                                       height_ratios=[1, 0.07,
                                                      1, 0.07,
                                                      1, 0.07,
                                                      1, 0.07,
                                                      1, 0.07])
case = '0_300_200_50'  # Vogels & Abbott 2005 results
fetch_parameter(path, seed, case)
axs, fr, isis, cvs, m_avg, talpha = plot_summary(gsy, path, seed, case)
for ax in axs[0::2]:
    ax.text(1, -0.3, s='Inhibitory (ns)',
            transform=ax.transAxes,
            va='center', ha='center', clip_on=False)
ax1 = axs[0]
draw_external_ip_line(axs[0::2], poi=True, text=True)
draw_external_ip_line(axs[1::2], poi=False, text=True)

plot_colorbars(gs, fr, isis, cvs, m_avg, talpha)
gs.tight_layout(fig, rect=[0, 0., 1, 1])

# plt.show()
ax0.text(1.1, 1.4, s='With metabolic current', transform=ax0.transAxes,
         color='k', va='center', ha='center', clip_on=False)

ax1.text(1.1, 1.4, s='Vogels & Abbott 2005', transform=ax1.transAxes,
         color='k', va='center', ha='center', clip_on=False)
plt.savefig('Figure3_nw_supp.png', dpi=300)
