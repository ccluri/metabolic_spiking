from math import exp, log


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

# # ROS definition
ros_inf = lambda atp, psi : ((atp*psi) + ((1-atp)*(1-psi)))**3
ros_tau = lambda atp, bls : 0.01 + 1200/(1*exp((log(bls) - 3.9)/0.25) + 1*exp(-(log(bls) - 4.6)/0.25))  # ros_tau1
ros_tau_slow = lambda atp, bls : 0.01 + 12000/(1*exp((log(bls) - 3.9)/0.25) + 1*exp(-(log(bls) - 4.6)/0.25))  # ros_tau_slow
ros_tau_fast = lambda atp, bls : 0.01 + 120/(1*exp((log(bls) - 3.9)/0.25) + 1*exp(-(log(bls) - 4.6)/0.25))  # ros_tau_fast
minisog_inf = lambda atp, psi : ((atp*psi) + ((1-atp)*(1-psi)))**3 + 0.1
aox_inf = lambda atp, psi : (((atp*psi)/1.15) + (((1-atp)*(1-psi))/1.15))**3
rotenone_inf = lambda atp, psi : (((atp*psi)/1.25) + (((1-atp)*(1-psi))/0.8))**3


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


def get_parkinsons_rotenone():
    # The effect is due to RET ROS collapse
    ros = Gate('park', 1, inf=rotenone_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros


get_parkinsons_type2 = get_ros
# The effect is due to massive inc in Q (ATP-per-spike)


def get_mini():
    ros = Gate('hyp', 1, inf=minisog_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros


def get_aox():
    ros = Gate('hyp', 1, inf=aox_inf, tau=ros_tau)
    ros.init_val(1, 0)
    return ros
