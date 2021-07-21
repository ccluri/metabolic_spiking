import numpy as np
import figure_properties as fp
import matplotlib.pyplot as plt
from mitochondria import Mito, to_dpsi
from scipy import interpolate
# from palettable.colorbrewer.qualitative import Set2_8 
# from cycler import cycler
from utils import Recorder

# def plot_bl_curve(ax):
#     atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
#     m = ax.plot(atp(30), psi(30), marker='*', markersize=25, c='k',
#                 markeredgewidth=1.5, markeredgecolor='k')
#     m = ax.plot(atp(150), psi(150), marker='*', markersize=25, c='gold',
#                 markeredgewidth=1.5, markeredgecolor='k')
    
    
# def excursion(ax2):
#     plot_bl_curve(ax2)
#     ax2.set_aspect('equal')
#     ax2.set_xlim(0, 1.)
#     ax2.set_ylim(0, 1.)
#     ax2.set_xlabel('$ATP_M$')
#     ax2.set_ylabel('$\Delta\psi_M$')
#     ax2.set_title('Respiratory state space')
#     return ax2


# def clean_ax(axs, baseline_atps, colors):
#     for ii, ax1 in enumerate(axs):
#         ax1.set_xscale('log')
#         ax1.set_xlabel('Baseline $ATP_c$ usage')
#         if ii < 2:
#             ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.5, 0.01))
#         else:
#             ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.55, 0.01))
#         for bl, cl, ml in zip(test_bl_atps, test_bl_clrs, test_bl_markers):
#             if ii<2:
#                 ax1.plot(bl, 0, marker=ml, clip_on=False, color=cl, markersize=10)
#             else:
#                 ax1.plot(bl, -49, marker=ml, clip_on=False, color=cl, markersize=10)
#         ax1.spines['top'].set_visible(False)
#         ax1.spines['right'].set_visible(False)
#     return


# def clean_ax2(axs, baseline_atps, colors):
#     for ii, ax1 in enumerate(axs):
#         ax1.set_xscale('log')
#         ax1.set_xlabel(r'$ATP_C \rightarrow ADP_C$ (/ms)'+'\n\n')
#         if ii < 2:
#             ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.5, 0.01))
#         else:
#             ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.55, 0.01))
#         ax1.spines['top'].set_visible(False)
#         ax1.spines['right'].set_visible(False)
#     return 
        

# def figure_steady_state(baseline_atps, gs, colors):
#     ax0 = plt.subplot(gs[0, 0])
#     times, r_mito = compute_trans_vals(30)
#     ax0.plot(times, r_mito.out['nad'], color=def_colors['nad'], lw=2, label='$NAD_m$')
#     ax0.plot(times, r_mito.out['atp'], color=def_colors['atp'], lw=2, label='$ATP_m$')
#     ax0.plot(times, r_mito.out['psi'], color=def_colors['psi'], lw=2, label='$\Delta\psi_m$')
#     ax0.plot(times, r_mito.out['pyr'], color=def_colors['pyr'], lw=2, label='$Pyr_m$')
#     ax0.plot(times, r_mito.out['cit'], color=def_colors['cit'], lw=2, label='$Cit_m$')
#     ax0.spines['top'].set_visible(False)
#     ax0.spines['right'].set_visible(False)
#     ax0.set_ylabel('(a.u.)')
#     ax0.set_xlabel('Time (ms)')
#     ax0.set_xlim(-10, 250)
#     ax0.legend(frameon=False, loc=8, ncol=2)
#     ax0.set_title('Baseline $ATP_c$ usage (30)')
    
#     ax1 = plt.subplot(gs[0, 1])
#     atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
#     bls = np.geomspace(1, 1000, 100)
#     color_cycle = cycler(color=Set2_8.hex_colors)
#     ax1.set_prop_cycle(color_cycle)
#     ax1.plot(bls, atp(bls), label=r'$ATP_m$', lw=2)
#     ax1.plot(bls, psi(bls), label=r'$\Delta\psi_m$', lw=2)
#     ax1.set_ylabel('(a.u.)')
#     ax1.set_title('Metabolic state')
#     ax1.set_ylim(0, 1)

#     ax2 = plt.subplot(gs[0, 2])
#     color_cycle = cycler(color=Set2_8.hex_colors[6:])
#     ax2.set_prop_cycle(color_cycle)
#     ax2.plot(bls, nad(bls), label=r'$NAD_m$', lw=2)
#     ax2.plot(bls, pyr(bls), label=r'$Pyr_m$', lw=2)
#     ax2.set_ylabel('(a.u.)')
#     ax2.set_title('Substrate concentration')
#     ax2.set_ylim(0, 1.8)
    
#     ax3 = plt.subplot(gs[0, 3])
#     color_cycle = cycler(color=Set2_8.hex_colors[2:])
#     ax3.set_prop_cycle(color_cycle)
#     ax3.plot(bls, vant(bls), label='$V_{ANT}$', lw=2)
#     ax3.plot(bls, vatp(bls), label='$V_{ATP}$', lw=2)
#     ax3.plot(bls, vresp(bls), label='$V_{RESP}$', lw=2)
#     ax3.set_ylabel('$(\mu M/s)$')
#     ax3.set_title('Reaction rates')
#     ax3.set_ylim(-49, 380)
#     clean_ax([ax1, ax2, ax3], baseline_atps, colors)
#     # ros = lambda atp, psi : atp*psi + ((1-atp)*(1-psi))
#     # ax1.plot(bls, ros(atp(bls), psi(bls)), label='ROS', lw=2)
#     return gs


def get_steady_state():
    try:
        kk = np.load('steady_state/steady_state.npz')
        baselines = kk['baselines']
        all_nads = kk['nad_ss']
        all_atps = kk['atp_ss']
        all_psis = kk['psi_ss']
        all_vant = kk['v_ant_ss']
        all_vatp = kk['v_atp_ss']
        all_vresp = kk['v_resp_ss']
        all_pyr = kk['pyr_ss']
        
        nad = interpolate.interp1d(baselines, all_nads, kind='cubic')
        atp = interpolate.interp1d(baselines, all_atps, kind='cubic')
        psi = interpolate.interp1d(baselines, all_psis, kind='cubic')
        vant = interpolate.interp1d(baselines, all_vant, kind='cubic')
        vatp = interpolate.interp1d(baselines, all_vatp, kind='cubic')
        vresp = interpolate.interp1d(baselines, all_vresp, kind='cubic')
        pyr = interpolate.interp1d(baselines, all_pyr, kind='cubic')
        
        return atp, psi, nad, pyr, vant, vatp, vresp
    except FileNotFoundError:
        print('Run compute_ss_vals with save=True')


def compute_trans_vals(baseline_atp):
    mi = Mito(baseline_atp=baseline_atp, dpsi=0.9)
    time = 2500
    dt = 0.01
    times = np.arange(0, time, dt)
    r_mito = Recorder(mi, ['pyr', 'cit', 'psi', 'atp', 'nad'], time, dt)
    for ii in range(len(times)):
        mi.update_vals(dt, atp_cost=0, leak_cost=0)
        r_mito.update(ii)
    return times, r_mito


def compute_ss_vals(save=False, k_r=None):
    # baseline steadystate_vals
    baselines = np.geomspace(1, 1000, 100)
    ss_atp = np.zeros_like(baselines)
    ss_psi = np.zeros_like(baselines)
    ss_nad = np.zeros_like(baselines)
    ss_pyr = np.zeros_like(baselines)
    ss_acc = np.zeros_like(baselines)
    ss_cit = np.zeros_like(baselines)
    ss_akg = np.zeros_like(baselines)
    ss_oaa = np.zeros_like(baselines)

    ss_v_ant = np.zeros_like(baselines)
    ss_v_atp = np.zeros_like(baselines)
    ss_v_resp = np.zeros_like(baselines)

    for kk, baseline_atp in enumerate(baselines):
        if k_r is None:
            m = Mito(baseline_atp=baseline_atp)
        else:
            m = Mito(baseline_atp=baseline_atp, k_r=k_r)
        m.steadystate_vals(time=2000)
        v_atp, v_ant, v_resp = m.fetch_actual_rates()
        ss_atp[kk] = m.atp
        ss_psi[kk] = m.psi
        ss_nad[kk] = m.nad

        ss_pyr[kk] = m.pyr
        ss_acc[kk] = m.acc
        ss_cit[kk] = m.cit
        ss_akg[kk] = m.akg
        ss_oaa[kk] = m.oaa

        ss_v_atp[kk] = v_atp
        ss_v_ant[kk] = v_ant
        ss_v_resp[kk] = v_resp
    if save:
        np.savez('steady_state/steady_state.npz',
                 baselines=baselines,
                 nad_ss=ss_nad, psi_ss=ss_psi, atp_ss=ss_atp,
                 pyr_ss=ss_pyr, acc_ss=ss_acc, cit_ss=ss_cit,
                 akg_ss=ss_akg, oaa_ss=ss_oaa,
                 v_resp_ss=ss_v_resp, v_ant_ss=ss_v_ant, v_atp_ss=ss_v_atp)


def compute_clamp_vals(baseline_atp, nad_ss, save=False, k_r=None):
    ATPx = np.arange(0, 1.05, 0.05)
    PSIx = np.arange(0, 1.05, 0.05)
    nx = ATPx.size
    ny = PSIx.size
    ATP, PSI = np.meshgrid(ATPx, PSIx, indexing='ij')
    # DPSI = PSI * (0.2 / 0.15)
    DPSI = to_dpsi(PSI)  # ((PSI*(PSI_max-PSI_min)) + PSI_min) / 0.15
    V_ANT = np.zeros_like(ATP)
    V_RESP = np.zeros_like(ATP)
    V_ATP = np.zeros_like(ATP)
    DATPDT = np.zeros_like(ATP)
    DPSIDT = np.zeros_like(ATP)
    for i in range(nx):
        for j in range(ny):
            atp = ATP[i, j]
            dpsi = DPSI[i, j]
            if k_r is None:
                m = Mito(baseline_atp=baseline_atp, atp=atp, dpsi=dpsi,
                         nad=nad_ss(baseline_atp))
            else:
                m = Mito(baseline_atp=baseline_atp, atp=atp, dpsi=dpsi,
                         nad=nad_ss(baseline_atp), k_r=k_r)
            DATPDT[i, j] = m.datpdt
            DPSIDT[i, j] = m.ddpsidt
            v_atp, v_ant, v_resp = m.fetch_actual_rates()
            print(atp, dpsi, v_atp, v_resp, i, j)
            V_ANT[i, j] = v_ant
            V_ATP[i, j] = v_atp
            V_RESP[i, j] = v_resp
    if save:
        filename = 'reaction_rates/reaction_rates_baseline_'
        filename += str(baseline_atp) + '.npz'
        np.savez(filename,
                 ATPx=ATPx, PSIx=PSIx,
                 DATPDT=DATPDT, DPSIDT=DPSIDT,
                 V_ANT=V_ANT, V_ATP=V_ATP, V_RESP=V_RESP)


def pre_compute_values(ss_vals=True, reaction_rates=True):
    if ss_vals:
        compute_ss_vals(save=True)  # gets me ss NAD, ATP, PSI vals as npz
    if reaction_rates:
        values = get_steady_state()
        atp_ss, psi_ss, nad_ss, pyr_ss, vant_ss, vatp_ss, vresp_ss = values
        # for bl_atp in range(10, 1000, 10):
        for bl_atp in [30, 150]:
            # gets me v_resp v_ant as npz at atp/psi when nad is at ss
            compute_clamp_vals(baseline_atp=bl_atp, nad_ss=nad_ss, save=True)
            print('Done', bl_atp)

# pre_compute_values(ss_vals=True, reaction_rates=True, fname_add='minpsi132')
# pre_compute_values(ss_vals=True, reaction_rates=True, fname_add='minpsi120')
# pre_compute_values(ss_vals=True, reaction_rates=True)
# pre_compute_values(ss_vals=False, reaction_rates=True)
