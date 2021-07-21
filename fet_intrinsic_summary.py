import numpy as np

from mitochondria import Mito
from gates import get_ros
from utils import Recorder, Q_nak


def fet_run_sim(mito_baseline, spike_quanta, psi_fac=0.1e-4, refrac=6,
                iclamp=2, theta_fet=-0.025):
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
    print(isi_min)
    #  Spike costs
    qdur = 1000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, spike_quanta)
    qlen = len(this_q)
    #  Mitochondria
    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=1000)
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
    #  ROS states
    ros = get_ros()
    ros.init_val(mi.atp, mi.psi)
    # Init vars
    spike_expns = np.zeros_like(t)
    ros_vals = np.zeros_like(t)
    ms_vals = np.zeros_like(t)
    spikes = []  # fake spikes
    fire_mode = False
    spiked = False
    elapsed = 0
    for i in range(len(t)):
        mi.update_vals(dt,
                       atp_cost=spike_expns[i],
                       leak_cost=spike_expns[i]*psi_fac)
        ros.update_vals(dt, mi.atp, mi.psi, spike_expns[i]+mito_baseline)
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
                spike_expns[i:i+qlen] += this_q
            except ValueError:
                spike_expns[i:] += this_q[:len(spike_expns[i:])]
        else:
            if elapsed < max(isi_min, refrac) or msig < theta_fet:
                elapsed += dt
            else:
                spiked = False
        r_mito.update(i)
    print(spikes)
    return spikes, time, t, r_mito, ros_vals, ms_vals, i_stim


def process_spikes_fet(spikes):
    spikes = np.array(spikes)
    spikes = spikes[np.where(spikes > 100)]  # Ignore the first 100ms
    isi = np.diff(spikes)
    try:
        isi_min = np.min(isi)
        isi_max = np.max(isi)
        fr = 1000 / isi_max
        cv = np.std(isi)/np.mean(isi)  # From Vogels2005
        if np.isclose(cv, 0, rtol=1e-1):
            mode = 2  # Regular
        else:
            mode = 4  # Adapting
    except ValueError:  # no spikes
        isi_min = None
        isi_max = None
        fr = 0
        mode = 3  # 'Silent'
        cv = np.nan  # CV remains undefined
    return isi_min, isi_max, mode, cv, fr


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

    
def mega_run_fet(filename_prefix='', ros_baseline=False):
    test_spike_quants = np.logspace(-0.15, 1.75, 10)
    test_bls_vals = np.linspace(60, 160, 10)
    QQ = len(test_spike_quants)
    BB = len(test_bls_vals)
    isi_min_tot = np.zeros((QQ, BB))
    isi_max_tot = np.zeros((QQ, BB))
    netros_tot = np.zeros((QQ, BB))
    mode_tot = np.zeros((QQ, BB))
    cv_tot = np.zeros((QQ, BB))
    fr_tot = np.zeros((QQ, BB))
    if ros_baseline:
        theta_fet = -1
    else:
        theta_fet = -0.05
    for qq, spike_quanta in enumerate(test_spike_quants):
        for bb, mito_baseline in enumerate(test_bls_vals):
            iclamp = 2
            spikes, time, t, r_mito, ros_vals, ms_vals, i_stim = fet_run_sim(mito_baseline,
                                                                             spike_quanta,
                                                                             iclamp=iclamp,
                                                                             theta_fet=theta_fet)

            isi_min, isi_max, md, cv, fr = process_spikes_fet(spikes)
            netros = np.average(ros_vals[int(200/0.01):])
            print(isi_min, isi_max, netros, md, cv, fr)
            isi_min_tot[qq, bb] = isi_min
            isi_max_tot[qq, bb] = isi_max
            netros_tot[qq, bb] = netros
            mode_tot[qq, bb] = md
            cv_tot[qq, bb] = cv
            fr_tot[qq, bb] = fr
    np.savez('./spike_compensation/spike_compensate_summary_fet_' + filename_prefix + '.npz',
             isi_min=isi_min_tot, isi_max=isi_max_tot,
             cv=cv_tot, netros=netros_tot, mode=mode_tot, fr=fr_tot,
             spike_quanta=test_spike_quants, mito_baseline=test_bls_vals,
             i_stim=i_stim)
    

if __name__ == '__main__':
    filename_prefix = 'iclamp2'
    mega_run_fet(filename_prefix)
    filename_prefix = 'ros_baseline'
    mega_run_fet(filename_prefix, ros_baseline=True)

