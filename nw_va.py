import sys
import time
import pickle
import brian2 as br
from brian2 import ms, mV, Hz, second
import numpy as np

br.prefs.devices.cpp_standalone.openmp_threads = 4


def fetch_ou_input():
    ou_tau = 100*ms
    ou_sigma = 0.03
    ou_eq = '''dnoise/dt = -noise/ou_tau + ou_sigma*sqrt(2/ou_tau)*xi : 1'''
    poi_eq = '''noise : 1 (linked)
                rates=noise*int(noise>0)/ms : Hz '''
    num_ous = 10
    OU_In = br.NeuronGroup(num_ous, ou_eq, method='euler')
    net_inp = br.NeuronGroup(total_neurons, poi_eq, refractory='5*ms',
                             threshold='rand()<rates*dt', method='euler')
    net_inp.noise = br.linked_var(OU_In, 'noise',
                                  index=np.repeat(np.arange(num_ous),
                                                  int(total_neurons/num_ous)))
    return net_inp


def fetch_poi_input():
    net_inp = br.NeuronGroup(total_neurons*0.8, 'rates : Hz', refractory='5*ms',
                             threshold='rand()<rates*dt', method='euler')
    net_inp.rates = 3*Hz
    return net_inp


def fetch_poi_onoff_input(sim_time):
    stim = br.TimedArray([3, 0] * Hz, dt=sim_time/2)
    net_inp = br.NeuronGroup(total_neurons, 'rates = stim(t) : Hz',
                             refractory='5*ms',
                             threshold='rand()<rates*dt', method='euler')
    return stim, net_inp


def main_loop(seed=30001, we=0.6, wi=6.7, A_max=15,
              tau_A=250, K_mu=500, K_sig=50, inp_type='poi_onoff', filename='',
              save_currs=False, awake=False, curr_idxs=None):
    # we, wi is unitless here, but to get nS, /100Mohm
    start_time = time.time()
    np.random.seed(seed)
    # Network parameters
    taum = 20*ms
    taue = 5*ms
    taui = 10*ms
    '''Tau A here is of the order of synaptic inputs - dictates change in
    baseline atp, or star. Not to be confused with the tau or ROS - that
    is of the order of spike durations'''
    tauA = tau_A*ms
    tauAmax = 1000*ms
    Vt = -50*mV
    Vr = -60*mV
    El = -60*mV
    E_ex = 0.*mV
    E_inh = -80.*mV
    I = 0*mV
    Amax = A_max*mV
    # Population's input
    if awake:
        if inp_type == 'ou':
            print('OU input')
            net_inp = fetch_ou_input()
        elif inp_type == 'poi':
            print('Poisson input')
            net_inp = fetch_poi_input()
        elif inp_type == 'poi_onoff':
            print('Poisson on(half-time)-off(half-time)')
            stim, net_inp = fetch_poi_onoff_input(sim_time)
    else:
        print('No input')
        inp_type = 'off'
    # Neuron model
    eqs = '''
    K : volt
    Ie = ge*(E_ex-v) : volt
    Ii = gi*(E_inh-v) : volt
    Isum = (Ie - Ii) : volt
    frac = (2*K/ (K+Isum)) : 1
    Minf = (2 / (1 + exp(-8*(frac-1.)))) -1: 1
    tauAf = tauAmax*exp(-(frac-1)**2/0.0098) + tauA :second
    dM/dt = (Minf - M) / tauAf : 1
    dv/dt  = (Ie+Ii-(v-El) + (Amax*M))*(1./taum) : volt (unless refractory)
    dge/dt = -ge/taue : 1
    dgi/dt = -gi/taui : 1
    '''
    total_exc = int(total_neurons*4 / 5)  # 3200
    total_inh = total_neurons - total_exc
    P = br.NeuronGroup(total_neurons, eqs, threshold='v>Vt',
                       refractory='5*ms - 3*ms*M',
                       method='euler',
                       reset='v=Vr; M-=0.1')
    P.v = (np.random.randn(len(P)) * 5. - 55.) * mV
    P.M = 0
    def_Ks = (K_sig*br.randn(len(P)) + K_mu) * mV
    P.K = def_Ks
    # Connections
    Ce = br.Synapses(P, P, on_pre='ge += we')
    Ci = br.Synapses(P, P, on_pre='gi += wi')
    Ce.connect('i<'+str(total_exc), p=0.02)
    Ci.connect('i>='+str(total_exc), p=0.02)
    if awake:
        ex_inp_conn = br.Synapses(net_inp, P, on_pre='''ge += we''')
        ex_inp_conn.connect(p=0.02)
    # Record
    s_mon = br.SpikeMonitor(P, ['M'])
    if save_currs:
        if not curr_idxs:
            t_mon = br.StateMonitor(P, variables=['Ie', 'Ii', 'M'],
                                    record=True)
            t_mon.active = False
            br.run(0.475*sim_time)
            t_mon.active = True
            br.run(0.1*sim_time)
            t_mon.active = False
            br.run(0.425*sim_time)
        else:
            t_mon = br.StateMonitor(P, variables=['Ie', 'Ii', 'M'],
                                    record=curr_idxs)
            br.run(sim_time)
    else:
        br.run(sim_time)
    print('Elapsed time', time.time() - start_time)
    if filename != '':
        data = s_mon.get_states(['t', 'i', 'M'])
        data['K'] = def_Ks
        with open('./netsim_results/' + filename + '_' + inp_type +
                  '_spks.pkl', 'wb') as ff:
            pickle.dump(data, ff)
        if save_currs:
            data = t_mon.get_states(['Ie', 'Ii', 'M', 't'])
            with open('./netsim_results/' + filename + '_' + inp_type +
                      '_currs.pkl', 'wb') as ff:
                pickle.dump(data, ff)
        print('Done dumping')
    return s_mon


def single_run(awake):
    br.device.reinit()
    br.device.activate()
    seed = 20
    we = 0.3
    wi = 5
    A_max = 25
    tau_A = 300
    K_mu = 200
    K_sig = 50
    # 20_0.3_5_25_300_200_50
    params = [seed, we, wi, A_max, tau_A, K_mu, K_sig]
    filename = 'nw_' + '_'.join([str(ii) for ii in params])
    main_loop(seed, we, wi, A_max, tau_A, K_mu, K_sig,
              save_currs=True, awake=awake, filename=filename,
              curr_idxs=[0, 1])
    return


total_neurons = 10000
sim_time = 10*second
if __name__ == '__main__':
    if sys.argv[-1] == 'sleep':
        single_run(awake=False)
    else:
        single_run(awake=True)
