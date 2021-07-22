import numpy as np
from scipy import interpolate

from mitochondria import Mito, to_dpsi
from utils import Recorder


def get_steady_state():
    try:
        kk = np.load('./steady_state/steady_state.npz')
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
        print('Run this file first')


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
        np.savez('./steady_state/steady_state.npz',
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
    DPSI = to_dpsi(PSI)
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
        filename = './reaction_rates/reaction_rates_baseline_'
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
        for bl_atp in [30, 150]:
            # gets me v_resp v_ant as npz at atp/psi when nad is at ss
            compute_clamp_vals(baseline_atp=bl_atp, nad_ss=nad_ss, save=True)
            print('Done', bl_atp)


if __name__ == '__main__':
    pre_compute_values(ss_vals=True, reaction_rates=True)

