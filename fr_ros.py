
import numpy as np
from gates import get_ros
from mitochondria import Mito
from utils import Recorder, Q_nak, add_arrow
import matplotlib.pyplot as plt
from figure_properties import *
from steady_state import get_steady_state
from matplotlib import gridspec, cm, patches
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib
# matplotlib.use('Agg')

def run_sim(mito_baseline, spike_quanta, psi_fac=0.1e-4, isi_min=3, adp_fac=0):
    print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
    print('Psi factor: ', psi_fac)
    dt = 0.01
    time = 2500
    t = np.arange(0, time, dt)
    qdur = 1000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, spike_quanta)
    qlen = len(this_q)
    ros = get_ros()

    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=1000)
    ros.init_val(1, 0)
    ros_thres = 0.16
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
    spike_expns = np.zeros_like(t)
    ros_vals = np.zeros_like(t)
    spikes = []  # fake spikes
    refracs = []
    fire_mode = False
    burst_mode = 0
    spiked = False
    elapsed = 0
    # isi_min = min(2+max(2*np.log10(spike_quanta), 0), 6)  # 5
    # isi_min = 2+max(5*np.log10(spike_quanta), 0)
    for i in range(len(t)):
        mi.update_vals(dt,
                       atp_cost=spike_expns[i],
                       leak_cost=spike_expns[i]*psi_fac)
        ros.update_vals(dt, mi.atp, mi.psi, spike_expns[i]+mito_baseline)
        # ros.update_vals(dt, mi.atp, mi.psi)
        ros_vals[i] = ros.get_val()
        if ros.get_val() > ros_thres and mi.atp > 0.5:  # RET ROS
            fire_mode = True
        else:
            burst_mode = 0  # stop bursting
            fire_mode = False
        if fire_mode and not spiked:
            spiked = True
            elapsed = 0
            burst_mode += 1  # perhaps a burst
            spikes.append(t[i])
            refracs.append(isi_min + (burst_mode*adp_fac))
            try:
                spike_expns[i:i+qlen] += this_q
            except ValueError:
                spike_expns[i:] += this_q[:len(spike_expns[i:])]
        else:
            if elapsed < isi_min + (burst_mode*adp_fac):
                elapsed += dt
            else:
                spiked = False
        r_mito.update(i)
    # Calulate instead using ATP and PSI
    print(spikes)
    return spikes, isi_min, time, ros_thres, t, r_mito, spike_expns, spike_quanta, ros_vals, refracs


def draw_dim_ln_x(ax, xs, y, text='', color='k', yscale=1):
    ax.plot(xs, [y, y], color=color, lw=0.5)
    ax.plot([xs[0], xs[0]], [yscale*(y - 0.002), yscale*(y + 0.002)], color=color, lw=0.5)
    ax.plot([xs[1], xs[1]], [yscale*(y - 0.002), yscale*(y + 0.002)], color=color, lw=0.5)
    ax.text(xs[1], y, s='  '+text, color=color)
    return ax

def make_bars_refrac(ax3, idx, spk, refracs, case, l_col):
    if idx % 2 == 0:
        sign = 1
    else:
        sign = -1
    for ix, sp in enumerate(spk):
        rect1 = patches.Rectangle(((sp-case.start)*100, sign),
                                  (case.isi_min)*100, sign, edgecolor=l_col, linewidth=0.5,
                                  facecolor=l_col)
        rect2 = patches.Rectangle(((sp-case.start+case.isi_min)*100, sign),
                                  (refracs[ix]-case.isi_min)*100, sign, linewidth=0.5,
                                  edgecolor =l_col, facecolor='none')
        ax3.add_patch(rect1)
        ax3.add_patch(rect2)
    return

def cases_considered(gs0, firing_tests):
    ax1 = plt.subplot(gs0[0, 2])  # ROS inset
    ax = plt.subplot(gs0[0, 1])  # ROS Signal
    ax2 = plt.subplot(gs0[1, 1])  # ATP Consumption
    ax3 = plt.subplot(gs0[1, 2])  # spikes
    axtext = plt.subplot(gs0[:, 0])
    for ii, case in enumerate(firing_tests):
        spk, isi_min, tt, ros_thres, t, r_mito, spike_expns, spike_quanta, ros_vals, refracs = run_sim(case.bl,
                                                                                                       case.q,
                                                                                                       isi_min=case.isi_min,
                                                                                                       adp_fac=case.adp_fac)
        if case.start is not False:
            start_idx = np.where(np.isclose(t, case.start))[0][0]
            end_idx = np.where(np.isclose(t, case.end))[0][0]
            l_curr = ax.plot(ros_vals[start_idx:end_idx], lw=0.5)
            l_col = l_curr[0].get_color()
            spk = np.array(spk)
            relevant_spikes = spk[np.where((spk >= case.start) &
                                           (spk <= case.end))]
            print(relevant_spikes)
            ros_at_spike = []
            for sp in relevant_spikes:
                ros_at_spike.append(ros_vals[np.where(t == sp)[0][0]])
            renorm_spikes = [(dd-case.start)*100 for dd in relevant_spikes]
            ax.plot(renorm_spikes, ros_at_spike, marker='o', c='k',
                    lw=0, markerfacecolor=l_col,
                    markersize=2, markeredgewidth=0.1)  # , markeredgecolor='k')
            
            ax1.plot(ros_vals[start_idx:end_idx], lw=0.5)
            ax1.plot(renorm_spikes, ros_at_spike, marker='o', c='k', lw=0,
                     markersize=2,  markerfacecolor=l_col, markeredgewidth=0.2)
            rect = patches.Rectangle((500, 0.157), 3500, 0.01, linewidth=0.5,
                                     edgecolor='gray', facecolor='none')
            ax.add_patch(rect)
            ax.set_ylabel('ROS (a.u)', labelpad=2)
            ax.set_ylim(*case.ros_ylim)
            ax.set_yticks(case.ros_ylim)
            ax.tick_params(axis="y", direction="in", pad=-15)
            # ax.set_yticks(*case.ros_ylim)
            ax.set_xlim(0, end_idx-start_idx)

            ax1.plot([0, 4000], [ros_thres, ros_thres], '--', c='gray', lw=0.5)
            ax1.set_xlim(500, 4000)
            ax1.text(0.7, 0.13, s='RET ROS Thres.', transform=ax1.transAxes,
                     ha='left', va='center', color='gray')
            ax1.set_ylim(0.157, 0.167)
            ax1.set_yticks([0.157, 0.167])
            ax1.set_ylabel('ROS (a.u.)', labelpad=2)
            # ax1.set_yticklabels(['0.157', '0.167'])
            # ax1.spines['left'].set_visible(False)
            ax1.tick_params(axis="y", direction="in", pad=-20)

            ax2.plot(case.bl + spike_expns[start_idx:end_idx], lw=0.5)
            ax2.set_ylim([ee+case.bl for ee in case.atp_ylim])
            ax2.set_yticks([])
            ax2.set_ylabel('ATP Cons.(/ms)', labelpad=2)
            ax2.set_yticklabels([])
            ax2.set_xlim(0, end_idx-start_idx)
            ax2.tick_params(axis="y", direction="in", pad=-15)
            if case.test_type == 'bl':
                ax2.plot(0, case.bl, marker='*', clip_on=False, color=l_col,
                         markersize=5, markeredgecolor='none', zorder=10)
                if ii % 2 == 1:
                    ax2.text(0+300, case.bl+case.atp_ylim[-1], s='+' + str(case.atp_ylim[-1]),
                         fontsize=5, ha='left', va='center')
            else:
                if ii % 2 == 0:
                    ax2.plot(0, case.bl, marker='*', clip_on=False, color='k',
                             markersize=5, markeredgecolor='none')
                    # ax2.plot(0, case.atp_ylim[-1], marker='*', clip_on=False,
                    #          color='k', markersize=5, markeredgecolor='none')
                if ii % 2 == 0:
                    ax2.text(0+300, case.bl+case.atp_ylim[-1], s='+' + str(case.atp_ylim[-1]),
                         fontsize=5, ha='left', va='center')
                    
            ymin, ymax = ax2.get_ybound()
            asb2 = AnchoredSizeBar(ax2.transData,
                                   int(50*100),
                                   '50 ms',
                                   loc=4,
                                   pad=0., borderpad=.0, sep=2,
                                   frameon=False, label_top=True,
                                   size_vertical=(ymax-ymin)/1000)
            ax2.add_artist(asb2)
            axtext.text(0., 0.5, s='Vs.', ha='left', color='k', va='center',
                        transform=axtext.transAxes, fontsize=5, clip_on=False)
            if ii % 2 == 0:
                axtext.text(0., 0.6, s=case.title_text, ha='left',
                            va='center', transform=axtext.transAxes,
                            color=l_col, fontsize=5)
            else:
                if len(case.title_text) > 13:
                    offset_pos = 0.2
                else:
                    offset_pos = 0
                axtext.text(0., 0.4-offset_pos, s=case.title_text,
                            ha='left', va='center',
                            transform=axtext.transAxes,
                            color=l_col, fontsize=5, clip_on=False)
            if ii % 2 == 0:
                y_offset = 1
            else:
                y_offset = -1
            all_spikes = []
            for rs in renorm_spikes:
                all_spikes.append([(rs, y_offset), (rs, 5*y_offset)])
            lc = LineCollection(all_spikes, colors=[l_col]*len(renorm_spikes), linewidths=1)
            ax3.add_collection(lc)
            # ax3.plot(renorm_spikes, [y_offset]*len(renorm_spikes), marker='o', c='k',
            #          lw=0, markerfacecolor=l_col,
            #          markersize=2, markeredgewidth=0.1)
            make_bars_refrac(ax3, ii, relevant_spikes, refracs, case, l_col)
            ax3.set_xlim(500, 4000)
            ax3.set_ylim(-8, 8)
            ax3.set_yticks([])
            ax3.set_yticklabels([])
            ax3.tick_params(axis="y", direction="in")
            ax3.text(-0.05, 0.5, s='Spikes', transform=ax3.transAxes)
            ymin, ymax = ax3.get_ybound()
            asb3 = AnchoredSizeBar(ax3.transData,
                                   int(5*100),
                                   '5 ms',
                                   loc=4,
                                   pad=0., borderpad=.0, sep=2,
                                   frameon=False, label_top=True,
                                   size_vertical=(ymax-ymin)/1000)
            ax3.add_artist(asb3)
            ax3.spines['left'].set_visible(False)
            clean_ax([axtext])
            axtext.spines['left'].set_visible(False)
            axtext.set_yticks([])
        else:
            ax.plot(t, ros_vals)
            ros_at_spike = []
            for sp in spk:
                ros_at_spike.append(ros_vals[np.where(t == sp)[0][0]])
            ax.plot(spk, ros_at_spike, 'ko')
            ax2.plot(t, spike_expns)

        clean_ax([ax, ax1, ax2, ax3])


def clean_ax(axs):
    for ax in axs:
        ax.set_xticks([])
        # ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

class TestBLCases(object):
    def __init__(self, bl, q, align_spike_at=None, isi_min=10,
                 adp_fac=2, event_t=None, title_text='',
                 ros_ylim=[0.12, 0.18], test_type='q',
                 atp_ylim=[0, 20]):
        self.bl = bl
        self.q = q
        if not align_spike_at:
            self.start = False
            self.event_t = False
        else:
            self.start = align_spike_at - 10
            self.event_t = event_t
        self.end = self.start + 200
        self.isi_min = isi_min
        self.adp_fac = max(adp_fac, 0)
        self.title_text = title_text
        self.ros_ylim = ros_ylim
        self.atp_ylim = atp_ylim
        self.test_type = test_type


figsize = cm_to_inches([8.3, 10])
# fig = plt.figure(figsize=(10, 10))
fig = plt.figure(figsize=figsize)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
gs = gridspec.GridSpec(4, 1, hspace=1)

# BL related cases
gs0 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
                                       height_ratios=[1, 1],
                                       width_ratios=[0.1, 0.25, 0.4],
                                       subplot_spec=gs[0, 0])
t_regular_slow = TestBLCases(38, 10, align_spike_at=833.72,
                             title_text=r'$\bigstar$'+'=38', test_type='bl',
                             ros_ylim=[0.11, 0.18], atp_ylim=[0, 20])
t_regular_fast = TestBLCases(35, 10, align_spike_at=914.64,
                             title_text=r'$\bigstar$'+'=35', test_type='bl',
                             ros_ylim=[0.11, 0.18], atp_ylim=[0, 20])
cases_considered(gs0, [t_regular_slow, t_regular_fast])

gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
                                       height_ratios=[1, 1],
                                       width_ratios=[0.1, 0.25, 0.4],
                                       subplot_spec=gs[1, 0])
t_regular_slow = TestBLCases(30, 10, isi_min=10,
                             align_spike_at=745.99,
                             test_type='bl', ros_ylim=[0.11, 0.18],
                             atp_ylim=[0, 50], title_text=r'$\bigstar$'+'=30')
                             #, event_t=936.19, , title_text='*=33')
                             # , ,
                             #  title_text='Q=6')
t_burster = TestBLCases(20, 10,
                        isi_min=7, adp_fac=1, align_spike_at=707.66,
                        test_type='bl', ros_ylim=[0.11, 0.18],
                        atp_ylim=[0, 50], title_text=r'$\bigstar$'+'=20\nlowered\nadaptation\n&\nrefractory')
                             #, event_t=797.4, title_text='*=30', test_type='bl')
                             # atp_ylim=[0, 14], ros_ylim=[0.11, 0.18],
                             # )
cases_considered(gs1, [t_regular_slow, t_burster])

l1 = matplotlib.lines.Line2D([0.13, 0.13], [0.53, 0.97], color='gray', lw=0.5,
                             transform=fig.transFigure, figure=fig)
tx1 = matplotlib.text.Text(0.05, 0.75, 'Q=10', ha='left', va='center',
                           transform=fig.transFigure, figure=fig)
l2 = matplotlib.lines.Line2D([0.1, 0.13], [0.75, 0.75], color='gray', lw=0.5,
                             transform=fig.transFigure, figure=fig)
fig.lines.extend([l1, l2])
fig.texts.extend([tx1])

gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
                                       height_ratios=[1, 1],
                                       width_ratios=[0.1, 0.25, 0.4],
                                       subplot_spec=gs[2, 0])
t_regular_slow = TestBLCases(35, 6, align_spike_at=715.46,
                             atp_ylim=[0, 14], ros_ylim=[0.13, 0.18],
                             event_t=827.38, title_text='Q=6')
t_burster_twin = TestBLCases(35, 5, align_spike_at=414.77,
                             atp_ylim=[0, 14], ros_ylim=[0.13, 0.18],
                             event_t=576.13, title_text='Q=5')
cases_considered(gs2, [t_regular_slow, t_burster_twin])

gs3 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
                                       height_ratios=[1, 1],
                                       width_ratios=[0.1, 0.25, 0.4],
                                       subplot_spec=gs[3, 0])
t_burster_triple = TestBLCases(35, 2, align_spike_at=323.54,
                               atp_ylim=[0, 14], ros_ylim=[0.14, 0.18],
                               event_t=442., title_text='Q=2')
# t_burster_bez_adp = TestBLCases(35, 2, align_spike_at=974.02,
#                                 adp_fac=0.1, event_t=1091.23)
t_burster_isi_low = TestBLCases(35, 2, align_spike_at=1568.92,
                                atp_ylim=[0, 14], ros_ylim=[0.14, 0.18],
                                adp_fac=0.1, event_t=1710.73,
                                isi_min=5, title_text='Q=2\nlowered\nadaptation\n&\nrefractory')
cases_considered(gs3, [t_burster_triple, t_burster_isi_low])

l3 = matplotlib.lines.Line2D([0.13, 0.13], [0.03, 0.47], color='gray', lw=0.5,
                             transform=fig.transFigure, figure=fig)
tx2 = matplotlib.text.Text(0.05, 0.25, r'$\bigstar$'+'=35', ha='left', va='center',
                           transform=fig.transFigure, figure=fig)
l4 = matplotlib.lines.Line2D([0.1, 0.13], [0.25, 0.25], color='gray', lw=0.5,
                             transform=fig.transFigure, figure=fig)
fig.lines.extend([l3, l4])
fig.texts.extend([tx2])


# # Q related cases only
# gs0 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
#                                        height_ratios=[1, 1],
#                                        width_ratios=[0.1, 0.25, 0.4],
#                                        subplot_spec=gs[0, 0])
# t_regular_slow = TestBLCases(35, 15, align_spike_at=523.15, event_t=717.2,
#                              title_text='Q=15')
# t_regular_fast = TestBLCases(35, 10, align_spike_at=601.1, event_t=757.87,
#                              title_text='Q=10')
# cases_considered(gs0, [t_regular_slow, t_regular_fast])

# gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
#                                        height_ratios=[1, 1],
#                                        width_ratios=[0.1, 0.25, 0.4],
#                                        subplot_spec=gs[1, 0])
# t_regular_slow = TestBLCases(35, 6, align_spike_at=715.46,
#                              atp_ylim=[0, 14], ros_ylim=[0.13, 0.18],
#                              event_t=827.38, title_text='Q=6')
# t_burster_twin = TestBLCases(35, 5, align_spike_at=414.77,
#                              atp_ylim=[0, 14], ros_ylim=[0.13, 0.18],
#                              event_t=576.13, title_text='Q=5')
# cases_considered(gs1, [t_regular_slow, t_burster_twin])

# gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
#                                        height_ratios=[1, 1],
#                                        width_ratios=[0.1, 0.25, 0.4],
#                                        subplot_spec=gs[2, 0])

# t_burster_triple = TestBLCases(35, 2, align_spike_at=323.54,
#                                atp_ylim=[0, 14], ros_ylim=[0.14, 0.18],
#                                event_t=442., title_text='Q=2')
# # t_burster_bez_adp = TestBLCases(35, 2, align_spike_at=974.02,
# #                                 adp_fac=0.1, event_t=1091.23)
# t_burster_isi_low = TestBLCases(35, 2, align_spike_at=1568.92,
#                                 atp_ylim=[0, 14], ros_ylim=[0.14, 0.18],
#                                 adp_fac=0.1, event_t=1710.73,
#                                 isi_min=5, title_text='Q=2\nlowered\nadaptation\n&\nrefractory')
# # cases_considered(gs2, [t_burster_triple, t_burster_bez_adp, t_burster_isi_low])
# cases_considered(gs2, [t_burster_triple, t_burster_isi_low])

# gs3 = gridspec.GridSpecFromSubplotSpec(2, 3, wspace=0.2, hspace=0.2,
#                                        height_ratios=[1, 1],
#                                        width_ratios=[0.1, 0.25, 0.4],
#                                        subplot_spec=gs[3, 0])
# # t_burster_triple = TestBLCases(35, 0.5)
# t_burster_bez_adp = TestBLCases(35, 0.5, adp_fac=0., isi_min=5,
#                                 atp_ylim=[0, 14], ros_ylim=[0.14, 0.18],
#                                 align_spike_at=252.9,
#                                 event_t=383.46000000000004, title_text='Q=0.5')
# t_burster_isi_low = TestBLCases(35, 0.5, adp_fac=0., isi_min=2,
#                                 atp_ylim=[0, 14], ros_ylim=[0.14, 0.18],
#                                 align_spike_at=417.73,
#                                 event_t=564.3100000000001, title_text='Q=0.5\nminimum\nadaptation\n&\nrefractory')
# cases_considered(gs3, [t_burster_bez_adp, t_burster_isi_low])

# # t_regular_slow = TestBLCases(35, 6, align_spike_at=715.46)
# # t_regular_fast = TestBLCases(35, 5, align_spike_at=414.77)


gs.tight_layout(fig)
# plt.savefig('fr_ros_q_cases.png', dpi=300)
plt.show()
