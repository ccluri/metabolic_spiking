import numpy as np
# import numba
# from numba import jit
from gates import Gate
from channel import Channel
from math import exp


class SNCDA(object):
    def __init__(self, name='test', **kwargs):
        # Non linear current, parameters
        print('SNcDA')
        self.v_reset = -72  # Reset point
        self.Q = kwargs.pop('Q', 20)  # cost per spike parameter
        
        # Leak, cell parameters
        self.E_L = kwargs.pop('E_L', -68)  # mV
        self.g_leak = kwargs.pop('g_leak', 1)
        self.tau_m = kwargs.pop('tau_m', 20)  # ms
        self.v = -68
        self.dvdt = 0
        
        # Mitochondria state
        self.atp = kwargs.pop('init_atp', 0.7)
        self.ros = kwargs.pop('init_ros', 0.)

        # Spiking properties
        self.vthr = kwargs.pop('vthr', -40)  # mostly always spikes
        self.refrc = kwargs.pop('refrc', 5)
        self.spiked = False
        self.elapse = 0
        print('Refractory period (ms):', self.refrc)
        
        # Nap
        self.g_nap = kwargs.pop('g_nap', 1.25)  # equivalent to c in Parga, Abbot paper
        self.E_Na = kwargs.pop('E_Na', 45)  # mV Rev. for Nap
        
        # K_ATP channels
        self.E_K = kwargs.pop('E_K', -70)  # mV
        self.g_katp = kwargs.pop('g_katp', 0)

        # Noise and stimulus
        self.i_inj = 0
        self.create_channels()

    def create_channels(self):
        # Leak
        self.Leak = Channel('Leak', self.g_leak, self.E_L, gates=[])

        # Persistent Sodium
        r_tau = lambda v: 5  # ms
        r_inf = lambda v: 1 / (1+exp((-v - 55)/2))
        p_tau = lambda v: 0.01
        p_inf = lambda v, ros: 1/(1+exp(-15*(ros-0.15)))
        r = Gate('r', 1, r_inf, r_tau)  # Modulates G_NaP
        p = Gate('p', 1, p_inf, p_tau)
        self.Na_P = Channel('Na_P', self.g_nap, self.E_Na, [p, r],
                            sp_gates=['p'])

        # # K-ATP channels - my own making
        # d_inf = lambda v, atp : (1/(1+exp(25*(atp-0.5))))*(1-(1/(1+exp(0.5*(v+78)))))
        # d_tau = lambda v : 250 # ms
        d_inf = lambda v : np.exp(-(v-self.E_K)/7)
        d_tau = lambda v: 5
        e_inf = lambda v, atp : (1/(1+exp(15*(atp-0.6))))
        e_tau = lambda v: 0.01
        d = Gate('d', 1, inf=d_inf, tau=d_tau)
        e = Gate('e', 1, inf=e_inf, tau=e_tau)
        self.K_ATP = Channel('K_ATP', self.g_katp, self.E_K, [e, d],
                             sp_gates=['e'])
        self.Leak.init_channel(self.v)
        self.Na_P.init_channel(self.v, self.ros)
        self.K_ATP.init_channel(self.v, self.atp)
        self.channels = [self.Leak, self.Na_P, self.K_ATP]
        
    def update_currs(self, dt):
        self.Leak.update_curr(dt, self.v)
        self.Na_P.update_curr(dt, self.v, self.ros*(self.atp > 0.5))
        self.K_ATP.update_curr(dt, self.v, self.atp)
        return
        
    def update_redox(self, atp, ros):  # Trigger from outside
        self.atp = atp  # Used to control K_ATP channels
        self.ros = ros  # Used to control self.P, self.T

    def update_vals(self, dt):
        self.update_currs(dt)
        I_leak = self.Leak.I
        I_nap = self.Na_P.I
        I_katp = self.K_ATP.I
        self.dvdt = (self.i_inj - I_leak - I_nap - I_katp) / self.tau_m
        self.v += self.dvdt*dt

        if not self.spiked and self.v >= self.vthr:
            self.spiked = True
            self.v = 20
            self.elapse = 0
            return True
        else:
            if self.elapse == dt:
                self.v = self.v_reset
                self.Na_P.gates[1].set_val(0.0)
            if self.elapse < self.refrc:
                self.elapse += dt
            else:
                self.spiked = False
            return False

    def set_iclamp(self, tt, clamp, hold=0):
        if tt < 50:
            self.i_inj = hold
        elif tt > 50 and tt < 55:
            self.i_inj = clamp
        else:
            self.i_inj = hold

            
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import Recorder
    from mitochondria import Mito
    from utils import Q_nak
    from gates import get_ros
    spike_quanta = 30
    baseline_atp = 25
    psi_fac = 0.1e-4
    mi = Mito(baseline_atp=baseline_atp)
    mi.steadystate_vals(time=1000)
    params = {'refrc': 5, 'Q': spike_quanta, 'init_atp': 1,
              'g_nap': 10, 'g_katp': 5}
    c1 = SNCDA('test', **params)
    dt = 0.01
    time = 2000
    qdur = 2000
    qtime = np.arange(0, qdur, dt)
    this_q = Q_nak(qtime, spike_quanta)
    qlen = len(this_q)
    ros = get_ros()
    ros.init_val(1, 0)
    t = np.arange(0, time, dt)
    ros_ss = np.arange(0, 1, dt/time)
    r_cell = Recorder(c1, ['v', 'i_inj', 'ros',
                           'Na_P.I',
                           'K_ATP.I',
                           'Leak.I',
                           'Na_P.gate_vals.r',
                           'Na_P.gate_vals.p'
                           ], time, dt)
    r_mito = Recorder(mi, ['atp', 'psi'], time, dt)
    spikes = []
    spike_expns = np.zeros_like(t)
    i_inj = np.zeros_like(t)
    t_start = 1000
    t_end = 1150
    i_inj[int(t_start/dt):int(t_end/dt)] = 50
    for i in range(len(t)):
        c1.i_inj = i_inj[i]
        mi.update_vals(dt,
                       atp_cost=spike_expns[i],
                       leak_cost=spike_expns[i]*psi_fac)
        ros.update_vals(dt, mi.atp, mi.psi, spike_expns[i]+baseline_atp)
        c1.update_redox(mi.atp, ros.val)
        spk = c1.update_vals(dt)
        if spk:
            print('SPIKE!')
            spikes.append(t[i])
            try:
                spike_expns[i:i+qlen] += this_q
            except ValueError:
                spike_expns[i:] += this_q[:len(spike_expns[i:])]
        r_cell.update(i)
        r_mito.update(i)

    plt.subplot(411)
    plt.plot(t, r_cell.out['v'])
    # plt.plot(t, r.output['dvdt'])
    plt.subplot(412)
    plt.plot(t, r_cell.out['i_inj'], label='inj')
    plt.plot(t, r_cell.out['Na_P.I'], label='Nap')
    plt.plot(t, r_cell.out['K_ATP.I'], label='KATP')
    plt.plot(t, r_cell.out['Leak.I'], label='Leak')
    plt.legend()
    plt.subplot(413)
    plt.plot(t, r_cell.out['Na_P.gate_vals.p'], label='P')
    plt.plot(t, r_cell.out['Na_P.gate_vals.r'], label='R')
    # plt.plot(t, ros_ss, label='ros')
    plt.legend()
    plt.subplot(414)
    plt.plot(t, r_mito.out['atp'], label='P')
    plt.plot(t, r_mito.out['psi'], label='R')
    plt.plot(t, r_cell.out['ros'], label='ROS')
    print(np.mean(r_cell.out['ros'][10000:]))
    plt.show()
