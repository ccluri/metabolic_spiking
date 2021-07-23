import pickle
import powerlaw

from itertools import groupby
from scipy.optimize import curve_fit
from brian2 import second, np


def process_data(times, nidx, M, bin_size):
    si_time = sim_time*1000  # ms but unitless
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
    fit = powerlaw.Fit(avalns, discrete=True, xmin=1)
    fit_t = powerlaw.Fit(avalns_t, discrete=True, xmin=1)
    try:
        popt, pcov = curve_fit(f_straight,
                               np.log10(avalns_t),
                               np.log10(avalns))
        print('Beta (fitted) :', popt[0])
        print('Intercept :', popt[1])
    except (ValueError, TypeError) as ee:
        return fit, fit_t, None
    return fit, fit_t, [popt[0], popt[1]]


def comp_powerlaw_coefs(fit, fit_t):
    alpha = fit.truncated_power_law.alpha
    s_cutoff = 1 / fit.truncated_power_law.Lambda
    print('Truncated Powerlaw, Alpha, S_Cutoff : ', alpha, s_cutoff)
    palpha = fit.power_law.alpha
    print('Powerlaw (size) Alpha : ', palpha)
    talpha = fit_t.power_law.alpha
    print('Powerlaw (time) Alpha : ', talpha)
    beta = (talpha-1)/(palpha-1)
    print('Beta (predicted) : ', beta)
    return alpha, s_cutoff, palpha, talpha, beta


def firing_prop_summary(total_neurons, times, nidx, duration):
    trains = invert_spikemonitor(total_neurons, times, nidx)
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
    return avg_fr, train_isi, cvs


def process_each_file(times, nidx, M, duration):
    avg_fr, train_isi, cvs = firing_prop_summary(total_neurons,
                                                 times, nidx, duration)
    bin_size = 1  # ms but unitless
    bins, valids, m_vals = process_data(times, nidx, M, bin_size)
    avalns, avalns_t, branch_fac = calc_avalan(valids, bin_size)
    fit, fit_t, beta_fit = compute_powerlaw_fits(avalns, avalns_t)
    print('All data included')
    alpha, s_cutoff, palpha, talpha, beta = comp_powerlaw_coefs(fit, fit_t)
    dict_entry = {'bins': bins,
                  'valids': valids,
                  'm_vals': m_vals,
                  'branch_fac': branch_fac,
                  'alpha': alpha,
                  's_cutoff': s_cutoff,
                  'palpha': palpha,
                  'talpha': talpha,
                  'fit_s': fit,
                  'fit_t': fit_t,
                  'beta': beta,
                  'beta_fit': beta_fit,
                  'avg_fr': avg_fr,
                  'train_isi': train_isi,
                  'cvs': cvs}
    return dict_entry


def invert_spikemonitor(total_neurons, times, nidx):
    s_mon_dict = {}
    for ii in range(total_neurons):
        s_mon_dict[ii] = np.sort(times[np.where(nidx == ii)])
    return s_mon_dict


def load_sim_file(ff):
    data = pickle.load(ff)
    times = data['t']*1000/second  # now in ms and unitless
    nidx = data['i']
    try:
        M = data['M']
    except KeyError:
        M = np.zeros_like(data['i'])
    return times, nidx, M


total_neurons = 10000
sim_time = 10  # sec
bin_size = 1  # ms
