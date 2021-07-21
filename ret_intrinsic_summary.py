import numpy as np

from mitochondria import Mito
from gates import get_ros
from utils import Recorder, Q_nak


def ret_run_sim(mito_baseline, spike_quanta, psi_fac=0.1e-4, refrac=6, theta_ret=0.025, tau_rise=0.6):
    print('Baseline : ', mito_baseline, 'Quanta :', spike_quanta)
    print('Psi factor: ', psi_fac, 'Refrac :', refrac)
    dt = 0.01
    time = 2500
    t = np.arange(0, time, dt)
    qdur = 1000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, spike_quanta, tau_rise=tau_rise)
    qlen = len(this_q)
    ros = get_ros()
    # Mitochondria
    mi = Mito(baseline_atp=mito_baseline)
    mi.steadystate_vals(time=1000)
    ros.init_val(mi.atp, mi.psi)
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
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
        if msig > theta_ret:  # RET ROS
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
            if elapsed < refrac:
                elapsed += dt
            else:
                spiked = False
        r_mito.update(i)
    print(spikes)
    return spikes, time, t, r_mito, ros_vals, ms_vals


def process_spikes(spikes, isi_preset, time):
    spikes = np.array(spikes)
    spikes = spikes[np.where(spikes > 500)]  # Ignore the first 500ms
    time = time-500
    isi = np.diff(spikes)
    try:
        isi_min = np.min(isi)
        isi_max = np.max(isi)
        fr = len(spikes) / (time)
        burst_array = np.isclose(isi, isi_preset, rtol=1e-1)
        bursting = np.any(burst_array)  # In maxfiring regime
    except ValueError:  # no spikes
        isi_min = None
        isi_max = None
        bursting = False
        fr = 0
    if bursting:
        cv = np.std(isi)/np.mean(isi)  # From Vogels2005
        isi_dur = isi_min
        try:
            ibi_dur = isi_max
            per_burst = np.argwhere(~burst_array)[-1][0] - \
                        np.argwhere(~burst_array)[-2][0]
            mode = 0  # 'Bursting'
        except IndexError:
            mode = 3  # 'Continuous'
            ibi_dur = time
            per_burst = int(time/isi_min)
    else:
        try:
            isi_dur = isi[-1]
            ibi_dur = isi[-1]
            per_burst = 1
            mode = 2  # 'Regular'
            cv = np.std(isi)/np.mean(isi)
        except IndexError:  # No spikes
            isi_dur = 0
            ibi_dur = 0
            per_burst = 0
            mode = 1  # 'Silent'
            cv = np.nan  # CV remains undefined
    return isi_dur, ibi_dur, per_burst, mode, cv, fr


def mega_run(filename_prefix='', refrac=6, tau_rise=0.6):
    test_spike_quants = np.logspace(-0.15, 1.75, 10)
    test_bls_vals = np.linspace(20, 40, 10)
    QQ = len(test_spike_quants)
    BB = len(test_bls_vals)
    isi_dur_tot = np.zeros((QQ, BB))
    ibi_dur_tot = np.zeros((QQ, BB))
    per_brst_tot = np.zeros((QQ, BB))
    netros_tot = np.zeros((QQ, BB))
    mode_tot = np.zeros((QQ, BB))
    cv_tot = np.zeros((QQ, BB))
    fr_tot = np.zeros((QQ, BB))
    for qq, spike_quanta in enumerate(test_spike_quants):
        for bb, mito_baseline in enumerate(test_bls_vals):
            spk, time, t, r_mito, ros_vals, ms_vals = ret_run_sim(mito_baseline,
                                                                  spike_quanta,
                                                                  refrac=refrac,
                                                                  tau_rise=tau_rise)
            isi_dur, ibi_dur, per_brst, md, cv, fr = process_spikes(spk, refrac, time)
            netros = np.average(ros_vals[int(20000):])
            print(isi_dur, ibi_dur, per_brst, netros, md, cv)
            isi_dur_tot[qq, bb] = isi_dur
            ibi_dur_tot[qq, bb] = ibi_dur
            per_brst_tot[qq, bb] = per_brst
            netros_tot[qq, bb] = netros
            mode_tot[qq, bb] = md
            cv_tot[qq, bb] = cv
            fr_tot[qq, bb] = fr
    np.savez('./spike_compensation/spike_compensate_summary_' + filename_prefix + '.npz',
             isi_dur=isi_dur_tot, ibi_dur=ibi_dur_tot, per_brst=per_brst_tot,
             cv=cv_tot, netros=netros_tot, mode=mode_tot, fr=fr_tot,
             spike_quanta=test_spike_quants, mito_baseline=test_bls_vals)


if __name__ == '__main__':
    for ref in [2, 6, 10]:
        for tau_r in [0.1, 0.6, 1.2]:
            filename_prefix = 'refrac_' + str(ref) + '_rise_' + format(tau_r, '.1f')
            mega_run(filename_prefix, refrac=ref, tau_rise=tau_r)
            print('Finished', filename_prefix)
