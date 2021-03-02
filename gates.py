import numpy as np
from math import exp, log
from utils import chann_colors, gate_styles


class Gate(object):
    def __init__(self, name, count, inf=None, tau=None,
                 alpha=None, beta=None):
        self.name = name
        self.count = count
        if alpha is not None:
            self.inf = lambda v: alpha(v) / (alpha(v) + beta(v))
            self.tau = lambda v: 1 / (alpha(v) + beta(v))
        else:
            self.inf = inf
            self.tau = tau
        self.val = 1.

    def init_val(self, v, s=None):
        if s is None:
            self.val = self.inf(v)
        else:
            self.val = self.inf(v, s)
        return self.val

    def dxdt(self, v, s=None, tau_s=None):
        if tau_s is None:
            tau = self.tau(v)
        else:
            tau = self.tau(v, tau_s)
        if s is None:
            return (self.inf(v) - self.val) / tau
        else:
            return (self.inf(v, s) - self.val) / tau

    def update_vals(self, dt, v, s=None, tau_s=None):
        self.val += self.dxdt(v, s, tau_s)*dt
        return self.val

    def set_val(self, val):
        self.val = val
        return
        
    def get_val(self):
        return self.val

    
# Defaults
# q10 = 2.3
qt = lambda celsius=22, q=2.3 : q**((celsius-22)/10)

# Hodgkin and Huxley channels
# http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
n_alpha = lambda v : (0.01 * (v + 55)) / (1 - exp(-0.1*(v+55)))
n_beta = lambda v : 0.125 * exp(-0.0125*(v+65))
m_alpha = lambda v : (0.1 * (v+40)) / (1-exp(-0.1*(v+40)))
m_beta = lambda v : 4.0 * exp(-0.0556*(v+65))
h_alpha = lambda v : 0.07 * exp(-0.05*(v+65))
h_beta = lambda v : 1.0 / (1 + exp(-0.1*(v+35)))


# K A type channnels
qq = qt(celsius=34, q=3)
a_inf = lambda v : (1. / (1 + exp(-(v + 31) / 6.)))**0.25
a_tau = lambda v : ((100. / (7 * exp((v+60) / 14.) + 29 * exp(-(v+60) / 24.))) + 0.1) / qq
b_inf = lambda v : 1. / (1 + exp((v + 66) / 7.))**0.5
b_tau = lambda v : ((1000. / (14 * exp((v+60) / 27.) + 29 * exp(-(v+60) / 24.))) + 1) / qq
c_inf = lambda v : 1. / (1 + exp((v + 66) / 7.))**0.5
c_tau = lambda v : ((90. / (1 + exp((-66-v) / 17.))) + 10) / qq
f_inf = lambda v, ros_i : (1/(1+exp(-15*(ros_i-0.65))))
f_tau = lambda v : 500 #ms 

# Ca - LVA (T-type) Avery and Johnston 1996, tau from Randall 1997
# shifted by -10 mv to correct for junction potential
# corrected rates using q10 = 2.3, target temperature 34, orginal 21
qw = qt(celsius=34, q=2.3)
t_inf = lambda v : 1/(1+exp(((v+30)-(-30))/-6))
t_tau = lambda v : (5+20/(1+exp(((v+30)-(-25))/5))) / qw

s_inf = lambda v : 1/(1+exp(((v+30)-(-80))/6.4))
s_tau = lambda v : (20+50/(1+exp(((v+30)-(-40))/7))) / qw
# ROS Action - my own making
q_inf = lambda v, atp : (1-(0.6/(1+exp(-40*(atp-0.6)))))
q_tau = lambda v : 300 #ms

# Ca - HVA (L-type) Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993
l_alpha = lambda v : (0.055*(-27-v))/(exp((-27-v)/3.8)-1)        
l_beta = lambda v : (0.94*exp((-75-v)/17))
r_alpha = lambda v : (0.000457*exp((-13-v)/50))
r_beta = lambda v : (0.0065/(exp((-v-15)/28)+1))

# # K-ATP channels - my own making
d_inf = lambda v, atp : (1/(1+exp(-25*(atp-0.5))))*(1-(1/(1+exp(-0.5*(v+58)))))
d_tau = lambda v : 30 # ms

# Na P channels and ROS - my own making
# ros_i is a ROS 
p_inf = lambda v, ros_i : (1/(1+exp(-50*(ros_i-0.2))))*(1-(1/(1+exp(-0.5*(v+50)))))
p_tau = lambda v : 6 # ms

# # ROS definition
ros_inf = lambda atp, psi : ((atp*psi) + ((1-atp)*(1-psi)))**3
# ros_tau = lambda atp : 1500/(1*exp((atp - 0.55)/0.07) + 1*exp(-(atp - 0.8)/0.07)) #
ros_tau = lambda atp, bls : 0.01 + 1200/(1*exp((log(bls) - 3.9)/0.25) + 1*exp(-(log(bls) - 4.6)/0.25))  # ros_tau1
ros_tau_2 = lambda atp, bls : 10000/(1*exp((log(bls) - 3.9)/0.1) + 1*exp(-(log(bls) - 4.6)/0.1))  # ros_tau2
ros_tau_slow = lambda atp, bls : 0.01 + 12000/(1*exp((log(bls) - 3.9)/0.25) + 1*exp(-(log(bls) - 4.6)/0.25))  # ros_tau_slow
ros_tau_fast = lambda atp, bls : 0.01 + 120/(1*exp((log(bls) - 3.9)/0.25) + 1*exp(-(log(bls) - 4.6)/0.25))  # ros_tau_fast

# 200*(1-(1/(1+exp(20*(atp-0.8))))) + 10  #ms
# drosdt = lambda atp, psi, ros : (ros_inf(atp, psi) - ros) / ros_tau(atp)   
hypoxia_inf = lambda atp, psi : (((atp*psi)/0.8) + (((1-atp)*(1-psi))/1.2))**3
minisog_inf = lambda atp, psi : ((atp*psi) + ((1-atp)*(1-psi)))**3 + 0.1
aox_inf = lambda atp, psi : (((atp*psi)/1.15) + (((1-atp)*(1-psi))/1.15))**3
teri_inf = lambda atp, psi : (((atp*psi)/1.05) + (((1-atp)*(1-psi))/1.05))**3
ageing_inf = lambda atp, psi : (((atp*psi)/1.5) + (((1-atp)*(1-psi))/1.0))**3

# LIF gates,
# Persistent Na conductivity gate
# PLIF_inf = lambda ros, atp : (atp > 0.4)/(1+exp(-30*(ros-0.35)))
PLIF_tau = lambda ros : 20 # ms
PLIF_inf = lambda ros, atp : 1/(1+exp(-10*(ros-0.25)))

# T-type Ca conductivity gate
# TLIF_inf = lambda ros, Q : 1/(1+exp(-10*((ros/Q)-0.4)))
TLIF_tau = lambda v : (20+50/(1+exp(((v+50))/3))) # ms
#TLIF_inf = lambda v, Q : 1*(1/(1+exp(((v+60))/-1)))/(1+np.exp(2*(Q-5)))
TLIF_inf = lambda v : 1*(1/(1+exp(((v+61))/-0.4)))

# # RAM definition
ram_inf = lambda cai : (1-(1/(1+exp(-0.05*((1e6*cai)-150))))) # cut off at 150 nm
ram_tau = 2000

def get_ros():
    ros = Gate('ros', 1, inf=ros_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros

def get_ros_slow():
    ros = Gate('ros_slow', 1, inf=ros_inf, tau=ros_tau_slow)
    ros.init_val(1, 0)
    return ros

def get_ros_fast():
    ros = Gate('ros_fast', 1, inf=ros_inf, tau=ros_tau_fast)
    ros.init_val(1, 0)
    return ros

def get_ros_2():
    ros = Gate('ros_2', 1, inf=ros_inf, tau=ros_tau_2)
    ros.init_val(1, 0)
    return ros

def get_parkinsons_type1():
    # The effect is due to RET ROS collapse
    ros = Gate('park', 1, inf=ageing_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros

get_parkinsons_type2 = get_ros
# The effect is due to massive inc in Q (ATP-per-spike)


def get_hypoxia():
    ros = Gate('hyp', 1, inf=hypoxia_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros

def get_teri():
    ros = Gate('hyp', 1, inf=teri_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros

def get_mini():
    ros = Gate('hyp', 1, inf=minisog_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros

def get_aox():
    ros = Gate('hyp', 1, inf=aox_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros

def get_fram(init_cai):
    fRAM = Gate('fRAM', 1, inf=ram_inf, tau=ram_tau)
    fRAM.init_val(init_cai)
    return fRAM

def get_lif_gates():
    P = Gate('P', 1, inf=PLIF_inf, tau=PLIF_tau)
    T = Gate('T', 1, inf=TLIF_inf, tau=TLIF_tau)
    #d = Gate('d', 2, inf=d_inf, tau=d_tau)
    t = Gate('t', 2, inf=t_inf, tau=t_tau)
    s = Gate('s', 1, inf=s_inf, tau=s_tau)
    channel_gate = {'P': [P], 'T': [T], 'Ca_T' : [t, s]}
#                    'K_ATP':[d]}
    all_gates = [P, T, t, s]
    all_chann = list(channel_gate.keys())
    return channel_gate, all_gates, all_chann

def get_gates():
    n = Gate('n', 4, alpha=n_alpha, beta=n_beta)
    m = Gate('m', 3, alpha=m_alpha, beta=m_beta)
    h = Gate('h', 1, alpha=h_alpha, beta=h_beta)
    a = Gate('a', 4, inf=a_inf, tau=a_tau)
    b = Gate('b', 1, inf=b_inf, tau=b_tau)
    c = Gate('c', 1, inf=c_inf, tau=c_tau)
    f = Gate('f', 1, inf=f_inf, tau=f_tau)
    t = Gate('t', 2, inf=t_inf, tau=t_tau)
    s = Gate('s', 1, inf=s_inf, tau=s_tau)
    q = Gate('q', 1, inf=q_inf, tau=q_tau)
    l = Gate('l', 2, alpha=l_alpha, beta=l_beta)
    r = Gate('r', 1, alpha=r_alpha, beta=r_beta)
    d = Gate('d', 2, inf=d_inf, tau=d_tau)
    p = Gate('p', 1, inf=p_inf, tau=p_tau)
    channel_gate = {'Na_T': [m, h], 'Na_P': [p],
                    'K_DR': [n], 'K_A': [a, b, c, f], 'K_ATP':[d],
                    'Ca_T' : [t, s, q], 'Ca_L': [l, r]}
    all_gates = [n, m, h, p, a, b, c, f, d, t, s, q, l, r]
    all_chann = list(channel_gate.keys())
    return channel_gate, all_gates, all_chann

def initialize_gates(v, atp=1, ros_i=0.3):
    init_vals = []
    channel_gate, all_gates, all_chann = get_gates()
    for gate in all_gates:
        if gate.name in ['q', 'p']:  # CaT and NaP
            init_vals.append(gate.init_val(v, ros_i))
        elif gate.name == 'd':  # K ATP
            init_vals.append(gate.init_val(v, atp))
        else:
            init_vals.append(gate.init_val(v))
    return init_vals

def fetch_inf_tau(gate, v, add_param=None):
    # v = np.arange(-100, 40, 0.7)
    ss = np.zeros_like(v)
    tt = np.zeros_like(v)
    for rr, vv in enumerate(v):
        if add_param is None:
            ss[rr] = gate.inf(vv)
        else:
            ss[rr] = gate.inf(vv, add_param)
        tt[rr] = gate.tau(vv)
    return ss, tt

def fetch_atp_ros(gate, atps, add_param=None):
    ss = np.zeros_like(atps)
    for rr, aa in enumerate(atps):
        ss[rr] = gate.inf(add_param, aa)
    return ss

def make_plots(chann, v, ax1, ax2=None, add_param=None):
    channel_gate, all_gates, all_chann = get_gates()
    for gate in channel_gate[chann]:
        try:
            ss, tt = fetch_inf_tau(gate, v)
        except TypeError:
            ss, tt = fetch_inf_tau(gate, v, add_param)
        if gate.name != 'f':
            ax1.plot(v, ss, linestyle=gate_styles[gate.name],
                     label='$'+gate.name+'_{\infty}$', lw=2, color=chann_colors[chann])
            if ax2 is not None:
                ax2.plot(v, tt, linestyle=gate_styles[gate.name],
                         label=r'$\tau_{'+gate.name+'}$', lw=2, color=chann_colors[chann])
    return ax1, ax2

def neat_plot(axs):
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        # ax.set_xticklabels([])
        # ax.axes.get_xaxis().set_visible(False)
        ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, 
                  borderaxespad=0, frameon=False)
        #ax.legend(frameon=False)
    return axs

def fix_ticks(axs, where, what):
    for ax in axs:
        ax.set_xticks(where)
        ax.set_xticklabels(what)
    return axs

def add_label(axs, label, x=True):
    for ax in axs:
        if x:
            ax.set_xlabel(label)
        else:
            ax.set_ylabel(label)
            ax.yaxis.set_label_coords(-0.15, 0.5)
    return axs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    # from figure_properties import *
    channel_gate, all_gates, all_chann = get_gates()
    v = np.arange(-100, 40, 0.7)    
    atps = np.arange(0, 1, 0.01)
    ross = np.arange(0, 1, 0.01)
    
    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(7, 3, hspace=0.75, wspace=0.35)
    # Na t
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax1, ax2 = make_plots('Na_T', v, ax1, ax2)

    # Na p
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax3, ax4 = make_plots('Na_P', v, ax3, ax4, add_param=1.) # ROS 1
    ss = fetch_atp_ros(channel_gate['Na_P'][0], ross, -65)
    ax5 = plt.subplot(gs[1, 2])
    ax5.plot(ross, ss, label='p (-65 mV)', lw=2, color=chann_colors['Na_P'])
    ax5.set_xlabel(r'$ROS_{SS}$')
    ax5.set_ylabel(r'$G_{Na_{P}}$')
    # K DR
    ax6 = plt.subplot(gs[2, 0])
    ax7 = plt.subplot(gs[2, 1])
    ax6, ax7 = make_plots('K_DR', v, ax6, ax7)
    # K A
    ax8 = plt.subplot(gs[3, 0]) 
    ax9 = plt.subplot(gs[3, 1])
    ax10 = plt.subplot(gs[3, 2])
    ax8, ax9 = make_plots('K_A', v, ax8, ax9, add_param=1) # ROS 1
    ross = np.arange(0, 1, 0.01)
    ss = fetch_atp_ros(channel_gate['K_A'][-1], ross, add_param=-65)
    #frac_inact = ross       # FIXXXXXXXX
    ax10.plot(ross, ss, label='f', lw=2, color=chann_colors['K_A'])
    ax10.set_xlabel(r'$ROS_{SS}$')
    ax10.set_ylabel('ac*(b*(1-f) + f)')
    #ax10.set_ylim(0, 1.05)
    # K ATP
    ax11 = plt.subplot(gs[4, 0])
    ax12 = plt.subplot(gs[4, 1])
    ax13 = plt.subplot(gs[4, 2])
    ax11, ax12 = make_plots('K_ATP', v, ax11, ax12, add_param=1)
    ss = fetch_atp_ros(channel_gate['K_ATP'][0], atps, add_param=-65)
    ax13.plot(atps, ss, label='d', lw=2, color=chann_colors['K_ATP'])
    ax13.set_xlabel(r'$ATP_{SS}$')
    ax13.set_ylabel(r'$G_{K_{ATP}}$')
    # Ca T
    ax14 = plt.subplot(gs[5, 0])
    ax15 = plt.subplot(gs[5, 1])
    ax16 = plt.subplot(gs[5, 2])
    ax14, ax15 = make_plots('Ca_T', v, ax14, ax15, add_param=1)
    ss = fetch_atp_ros(channel_gate['Ca_T'][-1], atps, add_param=-65)
    ax16.plot(atps, ss, linestyle='-.', label='q', lw=2, color=chann_colors['Ca_T'])
    ax16.set_xlabel(r'$ATP_{SS}$')
    ax16.set_ylabel(r'$G_{Ca_{T}}$')
    ax16.set_ylim(-0.05, 1.05)
    # Ca L
    ax17 = plt.subplot(gs[6, 0])
    ax18 = plt.subplot(gs[6, 1])
    ax17, ax18 = make_plots('Ca_L', v, ax17, ax18)

    all_axs = [eval('ax'+str(ii)) for ii in range(1, 19)]
    all_axs = neat_plot(all_axs)
    spl_axs = [eval('ax13'), eval('ax10'), eval('ax16'), eval('ax5')]
    non_spl = []
    inf_axs = [ax1, ax3, ax6, ax8, ax11, ax14, ax17]
    tau_axs = [ax2, ax4, ax7, ax9, ax12, ax15, ax18]
    for ii in all_axs:
        if ii not in spl_axs:
            non_spl.append(ii)
    all_axs = fix_ticks(non_spl, np.arange(-100, 60, 20), np.arange(-100, 60, 20)) 
    inf_axs = add_label(inf_axs, 'Steady state (a.u.)', x=False)
    # inf_axs = add_label(inf_axs, 'Memb. pot. (mV)', x=True)
    inf_axs = add_label([ax17, ax18], 'Membrane potential (mV)', x=True)
    tau_axs = add_label(tau_axs, 'Time constant (ms)', x=False)
    # tau_axs = add_label(tau_axs, 'Memb. pot. (mV)', x=True)
    # plt.savefig('ion_channels.png', dpi=150)
    plt.show()
