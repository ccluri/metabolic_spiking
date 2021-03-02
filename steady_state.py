import numpy as np
from figure_properties import *
import matplotlib.pyplot as plt
from mitochondria import Mito, to_dpsi
from gates import ros_inf
from scipy import interpolate
from palettable.colorbrewer.qualitative import Set2_8 
from cycler import cycler
from utils import Recorder
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

def figure_reaction_rate_stream(baseline_atp, ax,
                                fname_add='', color=None, norm=None):
    filename = 'reaction_rates/reaction_rates_baseline_'+str(baseline_atp)+fname_add+'.npz'
    print(filename)
    out = np.load(filename)
    ATPx = out['ATPx']
    PSIx = out['PSIx']
    # ATPx = np.arange(0, 1.05, 0.05)
    # PSIx = np.arange(0, 1.05, 0.05)
    # V_ANT = out.files['V_ANT']
    U = out['DATPDT']  # out['V_ATP']
    V = out['DPSIDT']  # out['V_RESP']
    speed = np.sqrt(U*U + V*V)
    print(baseline_atp, speed.max())
    # if norm == None:
    #     lw = 5*speed / speed.max()
    # else:
    #     lw = 5*speed / norm

    if color is not None:
        strm = ax.streamplot(ATPx, PSIx, U.T, V.T, density=0.6,
                             color=color, linewidth=lw.T)
    else:
        # strm = ax.streamplot(ATPx, PSIx, U.T, V.T, density=0.6,
        #                      color=speed.T, linewidth=2,
        #                      cmap='gray_r', norm=Normalize(vmin=0, vmax=speed.max()))
        strm = ax.contour(ATPx, PSIx, U.T)
        ax.clabel(strm, inline=1, fontsize=10)
        # if cax == None:
        #     plt.colorbar(strm.lines)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$ATP_m$')
    ax.set_ylabel('$\Delta\psi_m$')
    # ax.set_title('Rate of reaction (Baseline/$ATP_{Range}$ = ' + str(baseline_atp) + '/1000)')
    # ax.set_title('Rate of ATP generation')
    ax.set_title('Baseline $ATP_c$ usage ('+str(baseline_atp)+')')
    return ax, strm

def figure_reaction_rate_nullclines(baseline_atp, ax, fname_add=''):
    filename = 'reaction_rates/reaction_rates_baseline_'+str(baseline_atp)+fname_add+'.npz'
    print(filename)
    out = np.load(filename)
    ATPx = out['ATPx']
    PSIx = out['PSIx']
    U = out['DATPDT']
    V = out['DPSIDT']
    if baseline_atp == 30:
        manual_position = [(0.5, 0.45)]
    elif baseline_atp == 150:
        manual_position = [(0.75, 0.3)]
    strm_atp = ax.contour(ATPx, PSIx, U.T, levels=[0],
                          colors=[def_colors['atp']], linewidths=2, linestyles='dashdot')
    strm_psi = ax.contour(ATPx, PSIx, V.T, levels=[0],
                          colors=[def_colors['psi']], linewidths=2, linestyles='dashed')
    ax.clabel(strm_atp, inline=True, fontsize=20,
              fmt=r'$\frac{dATP_m}{dt}=0$')
    ax.clabel(strm_psi, inline=True, fontsize=20,
              fmt=r'$\frac{d\Delta\psi_m}{dt}=0$', manual=manual_position)

    # if cax == None:
        #     plt.colorbar(strm.lines)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$ATP_m$')
    ax.set_ylabel('$\Delta\psi_m$')
    # ax.set_title('Rate of reaction (Baseline/$ATP_{Range}$ = ' + str(baseline_atp) + '/1000)')
    # ax.set_title('Rate of ATP generation')
    ax.set_title(r'Rate constant $ATP_C \rightarrow ADP_C$'+'\n'+str(baseline_atp)+'/ms')
    return ax

def plot_bl_curve(ax):
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    m = ax.plot(atp(30), psi(30), marker='*', markersize=25, c='k',
                markeredgewidth=1.5, markeredgecolor='k')
    m = ax.plot(atp(150), psi(150), marker='*', markersize=25, c='gold',
                markeredgewidth=1.5, markeredgecolor='k')
    
    
def excursion(ax2):
    plot_bl_curve(ax2)
    ax2.set_aspect('equal')
    ax2.set_xlim(0, 1.)
    ax2.set_ylim(0, 1.)
    ax2.set_xlabel('$ATP_M$')
    ax2.set_ylabel('$\Delta\psi_M$')
    ax2.set_title('Respiratory state space')
    return ax2

def clean_ax(axs, baseline_atps, colors):
    for ii, ax1 in enumerate(axs):
        ax1.set_xscale('log')
        ax1.set_xlabel('Baseline $ATP_c$ usage')
        if ii < 2:
            ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.5, 0.01))
        else:
            ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.55, 0.01))
        for bl, cl, ml in zip(test_bl_atps, test_bl_clrs, test_bl_markers):
            if ii<2:
                ax1.plot(bl, 0, marker=ml, clip_on=False, color=cl, markersize=10)
            else:
                ax1.plot(bl, -49, marker=ml, clip_on=False, color=cl, markersize=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    return

def clean_ax2(axs, baseline_atps, colors):
    for ii, ax1 in enumerate(axs):
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$ATP_C \rightarrow ADP_C$ (/ms)'+'\n\n')
        if ii < 2:
            ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.5, 0.01))
        else:
            ax1.legend(loc=8, frameon=False, ncol=3, bbox_to_anchor=(.55, 0.01))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    return 
        

def figure_steady_state(baseline_atps, gs, colors):
    ax0 = plt.subplot(gs[0, 0])
    times, r_mito = compute_trans_vals(30)
    ax0.plot(times, r_mito.out['nad'], color=def_colors['nad'], lw=2, label='$NAD_m$')
    ax0.plot(times, r_mito.out['atp'], color=def_colors['atp'], lw=2, label='$ATP_m$')
    ax0.plot(times, r_mito.out['psi'], color=def_colors['psi'], lw=2, label='$\Delta\psi_m$')
    ax0.plot(times, r_mito.out['pyr'], color=def_colors['pyr'], lw=2, label='$Pyr_m$')
    ax0.plot(times, r_mito.out['cit'], color=def_colors['cit'], lw=2, label='$Cit_m$')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.set_ylabel('(a.u.)')
    ax0.set_xlabel('Time (ms)')
    ax0.set_xlim(-10, 250)
    ax0.legend(frameon=False, loc=8, ncol=2)
    ax0.set_title('Baseline $ATP_c$ usage (30)')
    
    ax1 = plt.subplot(gs[0, 1])
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    color_cycle = cycler(color=Set2_8.hex_colors)
    ax1.set_prop_cycle(color_cycle)
    ax1.plot(bls, atp(bls), label=r'$ATP_m$', lw=2)
    ax1.plot(bls, psi(bls), label=r'$\Delta\psi_m$', lw=2)
    ax1.set_ylabel('(a.u.)')
    ax1.set_title('Metabolic state')
    ax1.set_ylim(0, 1)

    ax2 = plt.subplot(gs[0, 2])
    color_cycle = cycler(color=Set2_8.hex_colors[6:])
    ax2.set_prop_cycle(color_cycle)
    ax2.plot(bls, nad(bls), label=r'$NAD_m$', lw=2)
    ax2.plot(bls, pyr(bls), label=r'$Pyr_m$', lw=2)
    ax2.set_ylabel('(a.u.)')
    ax2.set_title('Substrate concentration')
    ax2.set_ylim(0, 1.8)
    
    ax3 = plt.subplot(gs[0, 3])
    color_cycle = cycler(color=Set2_8.hex_colors[2:])
    ax3.set_prop_cycle(color_cycle)
    ax3.plot(bls, vant(bls), label='$V_{ANT}$', lw=2)
    ax3.plot(bls, vatp(bls), label='$V_{ATP}$', lw=2)
    ax3.plot(bls, vresp(bls), label='$V_{RESP}$', lw=2)
    ax3.set_ylabel('$(\mu M/s)$')
    ax3.set_title('Reaction rates')
    ax3.set_ylim(-49, 380)
    clean_ax([ax1, ax2, ax3], baseline_atps, colors)
    # ros = lambda atp, psi : atp*psi + ((1-atp)*(1-psi))
    # ax1.plot(bls, ros(atp(bls), psi(bls)), label='ROS', lw=2)
    return gs

def figure_steady_state_simpler(baseline_atps, gs, colors):
    # ax0 = plt.subplot(gs[0, 0])
    # times, r_mito = compute_trans_vals(30)
    # ax0.plot(times, r_mito.out['nad'], color=def_colors['nad'], lw=2, label='$NAD_m$')
    # ax0.plot(times, r_mito.out['atp'], color=def_colors['atp'], lw=2, label='$ATP_m$')
    # ax0.plot(times, r_mito.out['psi'], color=def_colors['psi'], lw=2, label='$\Delta\psi_m$')
    # ax0.plot(times, r_mito.out['pyr'], color=def_colors['pyr'], lw=2, label='$Pyr_m$')
    # ax0.plot(times, r_mito.out['cit'], color=def_colors['cit'], lw=2, label='$Cit_m$')
    # ax0.spines['top'].set_visible(False)
    # ax0.spines['right'].set_visible(False)
    # ax0.set_ylabel('(a.u.)')
    # ax0.set_xlabel('Time (ms)')
    # ax0.set_xlim(-10, 250)
    # ax0.legend(frameon=False, loc=8, ncol=2)
    # ax0.set_title('Baseline $ATP_c$ usage (30)')
    
    ax1 = plt.subplot(gs[0, 0])
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    bls = np.geomspace(1, 1000, 100)
    color_cycle = cycler(color=Set2_8.hex_colors)
    ax1.set_prop_cycle(color_cycle)
    ax1.plot(bls, atp(bls), label=r'$ATP_m$', lw=2)
    ax1.plot(bls, psi(bls), label=r'$\Delta\psi_m$', lw=2)
    ax1.set_ylabel('(a.u.)')
    ax1.set_title('Metabolic state')
    ax1.set_ylim(0, 1)

    ax2 = plt.subplot(gs[0, 1])
    color_cycle = cycler(color=Set2_8.hex_colors[6:])
    ax2.set_prop_cycle(color_cycle)
    ax2.plot(bls, nad(bls), label=r'$NAD_m$', lw=2)
    ax2.plot(bls, pyr(bls), label=r'$Pyr_m$', lw=2)
    ax2.set_ylabel('(a.u.)')
    ax2.set_title('Substrate concentration')
    ax2.set_ylim(0, 1.8)
    
    ax3 = plt.subplot(gs[0, 2])
    color_cycle = cycler(color=Set2_8.hex_colors[2:])
    ax3.set_prop_cycle(color_cycle)
    ax3.plot(bls, vant(bls), label='$V_{ANT}$', lw=2)
    ax3.plot(bls, vatp(bls), label='$V_{ATP}$', lw=2)
    ax3.plot(bls, vresp(bls), label='$V_{RESP}$', lw=2)
    ax3.set_ylabel('$(\mu M/s)$')
    ax3.set_title('Reaction rates')
    ax3.set_ylim(-49, 380)
    clean_ax2([ax1, ax2, ax3], baseline_atps, colors)
    # ros = lambda atp, psi : atp*psi + ((1-atp)*(1-psi))
    # ax1.plot(bls, ros(atp(bls), psi(bls)), label='ROS', lw=2)
    return gs

def get_steady_state(fname_add=''):
    try:
        kk = np.load('steady_state/steady_state'+fname_add+'.npz')
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

        
def compute_ss_vals(save=False, k_r=None, fname_add=''):
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
        if k_r == None:
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
        np.savez('steady_state/steady_state'+fname_add+'.npz', baselines=baselines,
                 nad_ss=ss_nad, psi_ss=ss_psi, atp_ss=ss_atp,
                 pyr_ss=ss_pyr, acc_ss=ss_acc, cit_ss=ss_cit,
                 akg_ss=ss_akg, oaa_ss=ss_oaa,
                 v_resp_ss=ss_v_resp, v_ant_ss=ss_v_ant, v_atp_ss=ss_v_atp)


def compute_clamp_vals(baseline_atp, nad_ss, save=False, k_r=None, fname_add=''):
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

            # FLAWED APPROACH HERE
            # ~~~~~~~~~~~~~~~~~~~
            # for t in tt:
                # m.update_vals(dt)
                # m.clamped_vals(dt, atp, dpsi)
            # m.clamped_vals()
            # ~~~~~~~~~~~~~~~~~~~
            DATPDT[i, j] = m.datpdt
            DPSIDT[i, j] = m.ddpsidt
            v_atp, v_ant, v_resp = m.fetch_actual_rates()
            print(atp, dpsi, v_atp, v_resp, i, j)
            V_ANT[i, j] = v_ant
            V_ATP[i, j] = v_atp
            V_RESP[i, j] = v_resp
    if save:
        filename = 'reaction_rates/reaction_rates_baseline_'+str(baseline_atp)+fname_add+'.npz'
        np.savez(filename,
                 ATPx=ATPx, PSIx=PSIx,
                 DATPDT=DATPDT, DPSIDT=DPSIDT,
                 V_ANT=V_ANT, V_ATP=V_ATP, V_RESP=V_RESP)

def pre_compute_values(ss_vals=True, reaction_rates=True):
    if ss_vals:
        compute_ss_vals(save=True) # gets me ss NAD, ATP, PSI vals as npz
    if reaction_rates:
        atp_ss, psi_ss, nad_ss, pyr_ss, vant_ss, vatp_ss, vresp_ss = get_steady_state()
        # for bl_atp in [10, 40, 60, 100, 1000]:
        for bl_atp in range(10, 1000, 10):
            # gets me v_resp v_ant as npz at atp/psi when nad is at ss
            compute_clamp_vals(baseline_atp=bl_atp, nad_ss=nad_ss, save=True)
            print('Done', bl_atp)

def pre_compute_values_inna(ss_vals=True, reaction_rates=True):
    'REDO as the clamp correction is not re-done'
    for k_r in [0.0012, 0.0016, 0.0018, 0.002]:
        sp_name = '_Inna_'+str(k_r)
        if ss_vals:
            compute_ss_vals(save=True, k_r=k_r, fname_add=sp_name)  # Inna
        if reaction_rates:
            atp_ss, psi_ss, nad_ss, pyr_ss, vant_ss, vatp_ss, vresp_ss = get_steady_state(fname_add=sp_name)
            compute_clamp_vals(baseline_atp=50, nad_ss=nad_ss,
                               save=True, k_r=k_r, fname_add=sp_name)
            print('Done', k_r)
            
# Figures for Inna 
def inna_figures(k_r):
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(121)
    ax = figure_reaction_rate_stream(baseline_atp=50, ax=ax, color='gray', norm=1300)
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    cntrlx, cntrly = atp(50), psi(50)
    print(ros_inf(cntrlx, cntrly), 'Control')
    ax.plot(cntrlx, cntrly, marker='*', markersize=10, color='gray')

    ax = figure_reaction_rate_stream(baseline_atp=50, ax=ax, norm=1300,
                                     color='#1f77b4', fname_add='_Inna_'+str(k_r))
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state(fname_add='_Inna_'+str(k_r))
    terix, teriy = atp(50), psi(50)
    print(ros_inf(terix, teriy), 'TERI')
    ax.plot(terix, teriy, marker='*', markersize=10, color='#1f77b4')

    # figure_steady_state(ax)
    # surf = figure_ros_scale(baseline_atp=40, ax=ax)
    ATP = np.arange(0, 1.05, 0.05)
    PSI = np.arange(0, 1.05, 0.05)
    ATP, PSI = np.meshgrid(ATP, PSI)
    ROS = (ATP*PSI) + ((1-ATP)*(1-PSI))

    ax2 = plt.subplot(122)
    surf = ax2.contourf(ATP, PSI, ROS, 30, cmap='coolwarm')
    ax2.plot(cntrlx, cntrly, marker='*', markersize=10, color='gray')
    ax2.plot(terix, teriy, marker='*', markersize=10, color='#1f77b4')
    ax2.set_aspect('equal')
    ax2.set_xlabel('ATP')
    ax2.set_ylabel('$\Delta\psi$')
    ax2.set_title('$ROS_{\infty}$')
    plt.show()

def ss_rr(baseline_atps, gs, colors):
    # fig = plt.figure(figsize=(10,5))
    atp, psi, nad, pyr, vant, vatp, vresp = get_steady_state()
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 2])]
    ax2 = excursion(plt.subplot(gs[0, 1]))
    for ii in [0, 1]:
        ax = axs[ii]
        # ax, strm = figure_reaction_rate_stream(baseline_atp=baseline_atps[ii],
        #                                        ax=ax, color=None, norm=1100)
        ax = figure_reaction_rate_nullclines(baseline_atp=baseline_atps[ii], ax=ax)
        cntrlx, cntrly = atp(baseline_atps[ii]), psi(baseline_atps[ii])
        ax.plot(cntrlx, cntrly, marker='*', markersize=25, color=colors[ii],
                markeredgecolor='k', markeredgewidth=1.5)
    # cbar = plt.colorbar(mappable=strm.lines, cax=plt.subplot(gs[0, -1]),
    #                     label='Reaction rate $\mu M/s$')

    
def make_reaction_rate_streams():
    # pre_compute_values(ss_vals=False)
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3,
                           wspace=0.3)
    ss_rr([30, 150],  gs, ['black', 'gold'])
    # plt.savefig('Reaction_rates_streams.png', dpi=300)
    #plt.savefig('Respiratory_statespace_nullclines.png', dpi=300)
    plt.show()

def make_steady_state_vals():    
    gs = gridspec.GridSpec(1, 4, wspace=0.3)
    fig = plt.figure(figsize=(25, 6))
    figure_steady_state(baseline_atps=test_bl_atps, gs=gs,
                        colors=test_bl_clrs)
    #plt.savefig('Steady_state_vals.png', dpi=300)
    plt.show()

def make_steady_state_vals_simpler():    
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    fig = plt.figure(figsize=(18, 6))
    figure_steady_state_simpler(baseline_atps=test_bl_atps, gs=gs,
                                colors=test_bl_clrs)
    plt.savefig('Steady_state_vals_Simpler.png', dpi=300)
    # plt.show()

# pre_compute_values(ss_vals=True, reaction_rates=True)
# pre_compute_values(ss_vals=False, reaction_rates=True)

# make_reaction_rate_streams()

# make_steady_state_vals_simpler()
# make_reaction_rate_streams()
