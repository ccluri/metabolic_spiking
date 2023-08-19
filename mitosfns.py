
import numpy as np
from mitochondria import Mito
from utils import Recorder, Q_nak
from gates import get_ros


def spike_quanta(baseline_atp, q, f_mcu=1e-3, ros=get_ros(), tau_Q=100, tot_time=500):
    '''Perturbation due to a spike'''
    dt = 0.01
    time = tot_time
    tt = np.arange(0, time, dt)
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # state 4 - wait till we reach here
    ros.init_val(m.atp, m.psi)
    Q_val = Q_nak(tt, 1, tau_Q=tau_Q, tau_rise=5)
    spike_val = np.zeros_like(tt) + m.atp
    leak_val = np.zeros_like(tt)
    ros_vals = np.zeros_like(tt)
    t_start = 150
    spike_val[int(t_start/dt):] -= Q_val[:len(spike_val[int(t_start/dt):])]*q
    leak_val[int(t_start/dt):] += Q_val[:len(leak_val[int(t_start/dt):])]*f_mcu*q
    rec_vars_list = ['atp', 'psi', 'k_ant', 'nad', 'pyr']
    m_record = Recorder(m, rec_vars_list, time, dt)
    for ii, ti in enumerate(tt):
        try:
            m.update_vals(dt, atp_cost=spike_val[ii],
                          leak_cost=leak_val[ii])
        except IndexError:
            m.update_vals(dt, leak_cost=0, atp_cost=m.atp)
        m_record.update(ii)
        ros.update_vals(dt, m.atp, m.psi,
                        m.k_ant*1000)   # keeping with the prev. ver.
        ros_vals[ii] = ros.get_val()
    return m_record, tt, ros_vals


def run_sim(test_freq, spike_quanta, psi_fac=1e-3, ros=get_ros(), tau_Q=100):
    fname_list = [test_freq, spike_quanta, psi_fac, ros.name, tau_Q]
    filename = '_'.join([str(yy) for yy in fname_list])
    filename += '.npz'
    try:
        kk = np.load('./spike_compensation/'+filename)
        bls, ros_metb_spikes = kk['bls'], kk['ros_metb_spikes']
        print('Previous run found, using file : ', filename)
    except FileNotFoundError:
        print('No prev compute found, running sim now')
        bls = np.geomspace(1, 1000, 20)
        ros_metb_spikes = np.zeros_like(bls)
        for ij, mito_baseline in enumerate(bls):
            print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
            print('Psi factor: ', psi_fac)
            dt = 0.01  # ms
            time = 10000  # ms
            t = np.arange(0, time, dt)
            qdur = 1000  # ms
            qtime = np.arange(0, qdur, dt)
            this_q = Q_nak(qtime, fact=1, tau_rise=5, tau_Q=tau_Q)
            qlen = len(this_q)
            mi = Mito(baseline_atp=mito_baseline)
            mi.steadystate_vals(time=1000)
            ros.init_val(mi.atp, mi.psi)
            spike_expns = np.zeros_like(t) + mi.atp
            leak_expns = np.zeros_like(t)
            test_isi = 1000 / test_freq
            test_isi_indx = int(test_isi / dt)
            num_spikes = int(time / test_isi)
            for sp in range(1, num_spikes+1):
                sp_idx = test_isi_indx*sp
                try:
                    spike_expns[sp_idx:sp_idx+qlen] -= this_q*spike_quanta
                    leak_expns[sp_idx:sp_idx+qlen] += this_q*psi_fac*spike_quanta
                except ValueError:
                    spike_expns[sp_idx:] -= this_q[:len(spike_expns[sp_idx:])]*spike_quanta
                    leak_expns[sp_idx:] += this_q[:len(leak_expns[sp_idx:])]*psi_fac*spike_quanta
            ros_vals = np.zeros_like(t)
            for i in range(len(t)):
                mi.update_vals(dt,
                               atp_cost=spike_expns[i],
                               leak_cost=leak_expns[i])
                ros.update_vals(dt, mi.atp, mi.psi,
                                mi.k_ant*1000)
                ros_vals[i] = ros.get_val()
            ros_metb_spikes[ij] = np.mean(ros_vals[int(-0.75*len(t)):])  # last 3/4 sample
        np.savez('./spike_compensation/'+filename,
                 bls=bls, ros_metb_spikes=ros_metb_spikes)
    return bls, ros_metb_spikes
