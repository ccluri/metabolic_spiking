import pickle
import powerlaw
import os
import glob

from itertools import groupby
from scipy.optimize import curve_fit
from brian2 import second, np, Hz

import matplotlib.pyplot as plt
from matplotlib import gridspec


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


def dump_avalan_summary(path):
    file_list = glob.glob(path+'/*_spikes.pkl')
    store_dict = {}
    for filepath in file_list:
        with open(filepath, 'rb') as ff:
            dict_key = filepath.split('/')[-1].rstrip('.pkl')
            times, nidx, M = load_sim_file(ff)
            dict_entry = process_each_file(times, nidx, M)
            store_dict[dict_key] = dict_entry
            folder_name_seed = './netsim_plots/' + dict_key.split('_')[0]
            if not os.path.isdir(folder_name_seed):
                os.mkdir(folder_name_seed)
            summary_spikes(total_neurons, sim_time, times, nidx, M,
                           dict_key, dict_entry,
                           filename='./'+folder_name_seed+'/'+dict_key+'.png')
    return store_dict


def summary_spikes(total_neurons, sim_time, times, nidx, M, dict_key,
                   dict_entry, filename=''):
    seed, we, wi, Amax, Atau, K_mu, K_sig = dict_key.split('_')
    plt.figure(figsize=(15, 12))  # width x height
    gs = gridspec.GridSpec(3, 4, wspace=0.3, hspace=0.2)  # row x column
    ax0 = plt.subplot(gs[0:2, :])
    ax3 = plt.subplot(gs[2, 0])
    ax4 = plt.subplot(gs[2, 1])
    ax5 = plt.subplot(gs[2, 2])
    ax6 = plt.subplot(gs[2, 3])
    ax0.scatter(times, nidx, c=M, alpha=1, s=0.05,
                marker='.', vmax=1, vmin=-1, cmap='viridis')
    ax0.set_title('mCOBA, Sim=' + dict_key + ', tau=' +
                  str(dict_entry['palpha'])[:4] + ', bf=' +
                  str(dict_entry['branch_fac'])[:4])
    ax0.set_ylabel('Neuron index')
    avg_fr = dict_entry['avg_fr']
    train_isi = dict_entry['train_isi']
    cvs = dict_entry['cvs']
    
    N, B = np.histogram(np.array(avg_fr), bins=np.linspace(0, 40, 41))
    ax3.bar(B[:-1], N / np.sum(N), width=np.diff(B), fill=False)
    ax3.plot((np.mean(avg_fr)*Hz, np.mean(avg_fr)*Hz),
             (0, np.max(N)/np.sum(N)), 'r', lw=2)
    ax3.set_xlabel('Avg firing rate (Hz)')

    N, B = np.histogram(train_isi, bins=np.logspace(0, 3.1, 30))
    ax4.bar(B[:-1], N / np.sum(N), width=np.diff(B), fill=False)
    ax4.plot((np.mean(train_isi), np.mean(train_isi)),
             (0, np.max(N)/np.sum(N)), 'r', lw=2)
    ax4.set_xscale('symlog')
    ax4.set_xlabel('ISI (ms)')

    N, B = np.histogram(cvs, bins=np.linspace(0, 3, 30))
    ax5.bar(B[:-1], N / np.sum(N), width=np.diff(B), fill=False)
    ax5.plot((np.mean(cvs), np.mean(cvs)), (0, np.max(N)/np.sum(N)), 'r', lw=2)
    ax5.set_xlabel('CV ISI')

    N, B = np.histogram(M, bins=np.linspace(-2, 1, 31))
    ax6.bar(B[:-1], N / np.sum(N), width=np.diff(B), fill=False)
    ax6.plot((np.mean(M), np.mean(M)), (0, np.max(N)/np.sum(N)), 'r', lw=2)
    ax6.set_xlabel('M')
    neat_axs([ax0, ax3, ax4, ax5, ax6])
    if len(filename) > 0:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def neat_axs(ax_list):
    for ax in ax_list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


total_neurons = 10000
sim_time = 10  # sec
bin_size = 1  # ms
