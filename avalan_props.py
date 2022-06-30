import sys
import glob
import pickle
import powerlaw

from itertools import groupby
from scipy.optimize import curve_fit
from brian2 import second, np, ms


def process_data(times, nidx, M, bin_size):
    si_time = sim_time/ms  # ms but unitless
    bins = np.arange(0, si_time, bin_size)
    valids = np.zeros_like(bins, dtype=int)
    m_vals = np.zeros_like(bins)
    for ii, jj in enumerate(bins):
        idxx = ((times > jj) & (times <= jj+bin_size))
        try:
            m_vals[ii] = np.sum(M[idxx])
        except ValueError:
            m_vals[ii] = 0
        valids[ii] = len(nidx[idxx])
    return bins, valids, m_vals


def calc_avalan(valids, bin_size):
    sigstr = []
    for ii in range(len(valids)-1):
        if valids[ii] != 0:
            sigstr.append(valids[ii+1] / valids[ii])
    branch_fac = np.mean(sigstr)
    print('Branching factor : ', branch_fac)
    all_avalan = [list(g) for k, g in groupby(valids, lambda x: x != 0) if k]
    avalns = [np.sum(ii) for ii in all_avalan]
    if bin_size < 1:
        print('Warning, log10 of values < 1 ms')
    avalns_t = [len(ii)*bin_size for ii in all_avalan]
    return avalns, avalns_t, branch_fac


def f_straight(x, A, B):
    return A*x + B


def compute_powerlaw_fits(avalns, avalns_t):
    avalns = np.array(avalns) + 1
    avalns_t = np.array(avalns_t) + 1
    try:
        popt, pcov = curve_fit(f_straight,
                               np.log10(avalns_t),
                               np.log10(avalns))
    except (ValueError, TypeError) as ee:
        return None, None, None
    fit = powerlaw.Fit(avalns, discrete=True, xmin=1)
    fit_t = powerlaw.Fit(avalns_t, discrete=True, xmin=1)
    print('Beta (fitted) :', popt[0])
    print('Intercept :', popt[1])
    return fit, fit_t, [popt[0], popt[1]]


def comp_powerlaw_coefs(fit, fit_t):
    try:
        alpha = fit.truncated_power_law.alpha
        s_cutoff = 1 / fit.truncated_power_law.Lambda
    except AttributeError:
        alpha = None
        s_cutoff = None
        palpha = None
    try:
        palpha = fit.power_law.alpha
        talpha = fit_t.power_law.alpha
        beta = (talpha-1)/(palpha-1)
    except AttributeError:
        palpha = None
        talpha = None
        beta = None
    print('Truncated Powerlaw, Alpha, S_Cutoff : ', alpha, s_cutoff)
    print('Powerlaw (size) Alpha : ', palpha)
    print('Powerlaw (time) Alpha : ', talpha)
    print('Beta (predicted) : ', beta)
    return alpha, s_cutoff, palpha, talpha, beta


def firing_prop_summary(total_neurons, times, nidx, duration, M):
    trains, Ms = invert_spikemonitor(total_neurons, times, nidx, M)
    avg_fr = []
    train_isi = []  # np.array(())
    cvs = []
    for ii in range(total_neurons):
        isi = np.diff(np.array(trains[ii]))
        train_isi.append(isi)
        # Units of duration is seconds as it reports Hz
        avg_fr.append(len(trains[ii]) / duration)
        if len(isi) > 0:
            cvs.append(np.std(isi)/np.mean(isi))
    return avg_fr, train_isi, cvs, Ms


def process_each_file(times, nidx, M, duration, full=True):
    avg_fr, train_isi, cvs, Ms = firing_prop_summary(total_neurons, times,
                                                     nidx, duration, M)
    bin_size = 1  # ms but unitless
    bins, valids, m_vals = process_data(times, nidx, M, bin_size)
    avalns, avalns_t, branch_fac = calc_avalan(valids, bin_size)
    print(avalns, avalns_t)
    fit, fit_t, beta_fit = compute_powerlaw_fits(avalns, avalns_t)
    print('All data included')
    alpha, s_cutoff, palpha, talpha, beta = comp_powerlaw_coefs(fit, fit_t)
    dict_entry = {'M_avg': np.mean(M),
                  'avalns': avalns,
                  'avalns_t': avalns_t,
                  'branch_fac': branch_fac,
                  'alpha': alpha,
                  's_cutoff': s_cutoff,
                  'palpha': palpha,
                  'talpha': talpha,
                  'beta': beta,
                  'beta_fit': beta_fit,
                  'avg_fr': avg_fr,
                  'train_isi': train_isi,
                  'Ms': Ms,
                  'cvs': cvs}
    if full:
        dict_entry.update({'m_vals': m_vals,
                           'bins': bins,
                           'valids': valids,
                           'fit_s': fit,
                           'fit_t': fit_t})
    return dict_entry


def invert_spikemonitor(total_neurons, times, nidx, M):
    s_mon_dict = {}
    m_mon_dict = {}
    for ii in range(total_neurons):
        X = times[np.where(nidx == ii)]
        Y = M[np.where(nidx == ii)]
        m_mon_dict[ii] = [pp for _, pp in sorted(zip(X, Y))]
        s_mon_dict[ii] = np.sort(X)
    return s_mon_dict, m_mon_dict


def load_sim_file(ff):
    data = pickle.load(ff)
    times = data['t']*1000/second  # now in ms and unitless
    nidx = data['i']
    try:
        M = data['M']
    except KeyError:
        M = np.zeros_like(data['i'])
    return times, nidx, M


def props_split(times, nidx, M, sim_time=10*second, full=True):
    hdur = sim_time/2
    fhalf_idx = np.where(times <= hdur/ms)
    times_fhalf = times[fhalf_idx]
    nidx_fhalf = nidx[fhalf_idx]
    M_fhalf = M[fhalf_idx]
    dict_fhalf = process_each_file(times_fhalf,
                                   nidx_fhalf,
                                   M_fhalf,
                                   duration=hdur,
                                   full=full)
    shalf_idx = np.where(times > hdur/ms)
    times_shalf = times[shalf_idx]
    nidx_shalf = nidx[shalf_idx]
    M_shalf = M[shalf_idx]
    dict_shalf = process_each_file(times_shalf,
                                   nidx_shalf,
                                   M_shalf,
                                   duration=hdur,
                                   full=full)
    return dict_fhalf, dict_shalf, M_fhalf, M_shalf


def dump_summary(path, seed, case='*'):
    file_list = glob.glob(path+'/nw_' + str(seed) + case + '_spks.pkl')
    if len(file_list) == 0:
        print('No matching simulations were found, exiting')
    for filepath in file_list:
        print(filepath)
        with open(filepath, 'rb') as ff:
            times, nidx, M, inet = load_sim_file(ff)
            d_fh, d_sh, M_fh, M_sh = props_split(times, nidx, M,
                                                 sim_time, full=False)
        dummyname = filepath.split('/')[-1].rstrip('_poi_onoff_spks.pkl')
        fname = dummyname.lstrip('nw_')
        with open(path + '/' + fname + '_summary.pkl', 'wb') as fx:
            pickle.dump((d_fh, d_sh), fx)
        print('Done computing for file matching: ', fname)
    return


total_neurons = 10000
sim_time = 10*second
bin_size = 1  # ms
if __name__ == '__main__':
    connectivity = 20
    path = './netsim_results/' + str(connectivity)
    seed = 20
    print(len(sys.argv))
    if sys.argv[-1] == 'vogels2005':
        print('Summaries for Vogels&Abbott2005')
        dump_summary(path, seed, '*_0_300_200_50_poi_onoff')
    elif sys.argv[-1] == 'metabolic':
        print('Summaries for metabolic current network')
        dump_summary(path, seed, '*_25_300_200_50_poi_onoff')
    else:
        if len(sys.argv) == 2:
            print('Summaries for files matching: ', sys.argv[-1])
            dump_summary(path, seed, sys.argv[-1])
        else:
            print('Invalid inputs')
            print('Allowed arguments: ')
            print('vogels2005, metabolic, (Valid)_spks.pkl')
            print('Here (Valid) can be a wildcard')
            
