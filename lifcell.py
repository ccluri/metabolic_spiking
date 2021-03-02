import numpy as np
# import numba
# from numba import jit
from gates import Gate
from channel import Channel
from math import exp

class LIFCell(object):
    def __init__(self, name='test', **kwargs):
        # Non linear current, parameters
        self.v1 = kwargs.pop('v1', -72)  # mV
        self.v2 = kwargs.pop('v2', self.v1 + 3)  # Reset point
        self.Q = kwargs.pop('Q', 20)  # cost per spike parameter
        
        # Leak, cell parameters
        self.E_L = kwargs.pop('E_L', -68)  # mV
        self.g_leak = kwargs.pop('g_leak', 1)
        self.tau_m = kwargs.pop('tau_m', 20)  # ms
        self.v = -68
        self.dvdt = 0
        
        # Mitochondria state
        self.atp = kwargs.pop('init_atp', 0.5)
        self.ros = kwargs.pop('init_ros', 0.)
        self.ros_ss = kwargs.pop('ros_ss', 0.1)  # depends on baseline firing

        # Spiking properties
        self.vthr = kwargs.pop('vthr', -40)  # mostly always spikes
        r_inf = lambda ros, Q: 1/((1+np.exp(-10*(ros-0.55)))*(1+np.exp(1.2*(Q-10))))
        self.refrc = kwargs.pop('refrc', 5 - 4*r_inf(self.ros_ss, self.Q))  # ms
        self.spiked = False
        self.elapse = 0
        print('Refractory period (ms):', self.refrc) 
        
        # Nap
        self.g_nap = kwargs.pop('g_nap', 1.25)  # equivalent to c in Parga, Abbot paper
        self.v3 = kwargs.pop('v3', 50) # mV Rev. for Nap
        # self.E_NaP = self.v3
        
        # Adaptation current parameters
        self.E_K = kwargs.pop('E_K', -80) # mV
        self.gka = 0 #.14 after each spike
        self.deltagka = kwargs.pop('deltagka', 0) # default is no adaptation
        # # A-type inacstivation - removal - implying adpatation relaxation
        ka_adapt = lambda ros : 1- (1/(1+np.exp(-15*(ros-0.35))))
        # self.tau_ka = kwargs.pop('tau_ka', 100) # ms
        self.tau_ka = 100*(ka_adapt(self.ros_ss))
        self.dgkadt = 0
        print('Time constant A-type adaptation :', self.tau_ka)
        
        # CaT
        self.g_cat = kwargs.pop('g_cat', 0.)
        q_inf = lambda ros, Q : 1/((1+np.exp(-10*(ros-0.35)))*(1+np.exp(0.2*(Q-18))))
        self.g_cat *= q_inf(self.ros_ss, self.Q)
        self.E_Ca = 125

        # BK type channel adaptation
        self.tau_bk = kwargs.pop('tau_bk', 100)
        self.gbk = 0
        self.deltagbk = kwargs.pop('deltagbk', 0)
        self.dgbkdt = 0
        
        # K_ATP channels
        self.g_katp = kwargs.pop('g_katp', 0)

        # Noise and stimulus
        self.noise = 0
        self.i_inj = 0

        self.create_channels()

    def create_channels(self):
        # Leak
        self.Leak = Channel('Leak', self.g_leak, self.E_L, gates=[])

        # Persistent Sodium
        # p_tau = lambda ros : 200 # ms
        # p_inf = lambda ros, atp : (atp>0.45)/((1+exp(-45*(ros-0.15)))) #*(1+np.exp(0.5*(self.Q-10))))

        r_tau = lambda v : 25 # ms
        r_inf = lambda v : 1/ (1+exp((-v-56 )/2))
        p_tau = lambda v : 100
        p_inf = lambda v, ros : 1/(1+exp(-25*(ros-0.15)))
        r = Gate('r', 1, r_inf, r_tau) # Modulates G_NaP
        p = Gate('p', 1, p_inf, p_tau)
        self.Na_P = Channel('Na_P', self.g_nap, self.v3, [p, r],
                            sp_gates=['p'])

        
        # Ca - LVA (T-type) 
        # A single column thalamocortical network model (Traub et al 2005)
        t_inf = lambda v : 1 / (1+exp((-v-56 ) / 6.2 ))
        t_tau = lambda v : 0.204 + 0.333 / (exp((v+15.8)/18.2) + exp((-v-131) /16.7))
        s_inf = lambda v : 1 / ( 1 + exp( ( v + 80 ) / 4 ) )
        s_tau = lambda v : 9.32 + 0.333 * exp( ( -v - 21 ) / 10.5 )
        

        tx = Gate('tx', 2, inf=t_inf, tau=t_tau)
        s = Gate('s', 1, inf=s_inf, tau=s_tau)
        # q_inf = lambda v, ros : 1/((1+np.exp(-40*(ros-0.3)))*(1+np.exp(0.1*(self.Q-7.5))))
        # q_tau = lambda v : 100
        # q = Gate('q', 0, inf=q_inf, tau=q_tau)
        self.Ca_T = Channel('Ca_T', self.g_cat, self.E_Ca, [tx, s]) #, q])

        # # K-ATP channels - my own making
        d_inf = lambda v, atp: (1/(1+exp(25*(atp-0.5))))*(1-(1/(1+exp(0.5*(v+78)))))
        d_tau = lambda v: 250  # ms
        d = Gate('d', 1, inf=d_inf, tau=d_tau)
        self.K_ATP = Channel('K_ATP', self.g_katp, self.E_K, [d],
                             sp_gates=['d'])
        

        self.Leak.init_channel(self.v)
        self.Na_P.init_channel(self.v, self.ros)
        self.Ca_T.init_channel(self.v, self.ros)
        self.K_ATP.init_channel(self.v, self.atp)
        self.channels = [self.Leak, self.Na_P, self.Ca_T, self.K_ATP]
        

    def update_currs(self, dt):
        self.Leak.update_curr(dt, self.v)
        self.Na_P.update_curr(dt, self.v, self.ros)
        self.Ca_T.update_curr(dt, self.v, self.ros)
        self.K_ATP.update_curr(dt, self.v, self.atp)
        self.v2 = min((-1*self.Ca_T.I/self.g_leak) + self.E_L, -45)
        # print(self.v2)
        # self.v2 = self.g_cat*
        return
        
    def update_redox(self, atp, ros):  # Trigger from outside
        self.atp = atp  # Used to control K_ATP channels
        self.ros = ros  # Used to control self.P, self.T

    def derivatives(self):
        channel_currs = 0
        for chann in self.channels:
            channel_currs += getattr(chann, 'I')
        self.dgkadt = -self.gka / self.tau_ka # faster adaptation due to KA type
        self.dgbkdt = -self.gbk / self.tau_bk # slower adaptation due to BK type
        self.dvdt = (self.i_inj
                     -self.noise
                     -(self.gka + self.gbk)*(self.v -self.E_K) # Adaptation
                     -channel_currs) / self.tau_m

        
    def update_vals(self, dt):
        self.update_currs(dt)
        self.derivatives()
        self.gka += self.dgkadt*dt
        self.gbk += self.dgbkdt*dt
        self.v += self.dvdt*dt

        if not self.spiked and self.v >= self.vthr:
            self.spiked = True
            # self.startT = True
            self.v = 20
            self.elapse = 0
            self.gka += self.deltagka
            self.gbk += self.deltagbk
            return True
        else:
            if self.elapse == dt:
                self.v = self.v2 - 2
                #self.Na_P.gates[0].set_val(0)
                self.Na_P.gates[1].set_val(0.0)
            if self.elapse < self.refrc:
                #self.Na_P.gates[0].set_val(0)
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

if __name__=='__main__':
    from utils import Recorder
    params = {'deltaga': 0, 'refrc': 5, 'Q':1, 'init_atp':0.5, 'g_nap':100.25, 'g_cat':1000}
    c = LIFCell('test', **params)
    dt = 0.01
    time = 1000
    t = np.arange(0, time, dt)
    ros_ss = np.arange(0, 1, dt/time)
    r = Recorder(c, ['v', 'i_inj',
                     'Na_P.I',
                     'Ca_T.I',
                     'Leak.I',
                     'Na_P.gate_vals.p',
                     'Ca_T.gate_vals.tx',
                     'Ca_T.gate_vals.s',
                     'Ca_T.gate_vals.q', 'v2'], time, dt)
    # print(getattr(c.K_DR.gates[0], 'm'))
    spikes = []
    
    for i in range(len(t)):
        tt = t[i]
        c.update_redox(0.5, 0.4)
        # c.set_iclamp(tt, clamp=50)

        spk = c.update_vals(dt)
        if spk : spikes.append(tt)
        r.update(i)

    #print(len(spikes), spikes)
    print(len(spikes))
    print(len(spikes)*1e3 / time, 'Hz')

    try:
        print(spikes[0])
    except IndexError:
        print('No spikes for you!')
        
    print(c.Na_P.gate_vals.p,
          c.Ca_T.gate_vals.q,
          c.Ca_T.I/c.g_cat,  'P', 'Q', 'V_Ca')
    print(c.Na_P.I, c.Ca_T.I, c.Leak.I)
    import matplotlib.pyplot as plt
    plt.subplot(311)
    plt.plot(t, r.out['v'])
    plt.plot(t, r.out['v2'])
    # plt.plot(t, r.output['dvdt'])
    plt.subplot(312)
    plt.plot(t, r.out['i_inj'], label='inj')
    plt.plot(t, r.out['Na_P.I'], label='Nap')
    plt.plot(t, r.out['Ca_T.I'], label='CaT')
    plt.plot(t, r.out['Leak.I'], label='Leak')
    plt.legend()
    plt.subplot(313)
    plt.plot(t, r.out['Na_P.gate_vals.p'], label='P')
    plt.plot(t, r.out['Ca_T.gate_vals.tx'], label='T')
    plt.plot(t, r.out['Ca_T.gate_vals.s'], label='S')
    plt.plot(t, r.out['Ca_T.gate_vals.q'], label='Q')

    # plt.plot(t, ros_ss, label='ros')
    plt.legend()
    plt.show()
