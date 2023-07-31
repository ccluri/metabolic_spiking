import numpy as np
from math import exp, erf

# -------------------------------------
# K ANT =  0.001; PSI_max = 0.1865
# K ANT =  0.005; PSI_max = 0.1833

# K ANT = 0.2; PSI_min = 0.132
# K ANT =  0.5; PSI_min = 0.1288
# K ANT = 1; PSI_min = 0.1273
# -------------------------------------

PSI_max = 0.190  # V
PSI_min = 0.125  # V

R = 8.314  # J/(mol.K)
T = 298  # K
F = 96465  # C/mol

a_m = 100  # /M
b_m = 4  # /M
at = 4.16e-3  # M
dg0 = 30540/R/T
frt = 1.2*3*F/R/T
h2opi = 1/2.4e-3

nt = 1.07e-3  # M (max NAD)
asp = 1.6e-3  # aspartate M
glu = 5.3e-3  # glutamate M
psim = 0.150  # V delta psi at steady state
C_mito = 6.75e-3  # mM/V
K_nad = 0.002  # M
Keq = 0.12*glu/asp
# print(Keq, 'K_equilibrium')
#  kant = 0.0538700
katp = 0.1318967  # M/s
kr = 0.0024905  # M/s

k1 = 38e-6  # M/s
k2 = 152  # /(Ms)
k3 = 57142  # /(Ms)
k4 = 53  # /(Ms)
k5 = 82361  # /(M^2 s)
k6 = 3.2e-3  # /s
k7 = 40  # /(Ms)
k8 = 3.6  # /s
kl = 0.426e-3  # M/(Vs)

pyrs = 0.14e-3  # M
accs = 0.07e-3  # M
cits = 0.4e-3  # M
kgs = 0.25e-3  # M
oaas = 0.005e-3  # M
nads = 0.94e-3  # M
atps = 3.23e-3  # M
psis = 0.150  # V

dtm = k1 / pyrs

beta2 = k2/k1*nt*pyrs
beta3 = k3/k1*accs*oaas
beta4 = k4/k1*cits*nt
beta5 = k5/k1*kgs*nt*at
beta6 = k6/k1*oaas
beta7 = k7/k1*pyrs*at
beta8 = k8/k1*oaas
betar = lambda kR=kr: kR/k1
bant = lambda k_ant: k_ant/k1*at
betal = lambda kL=kl: kL/k1*psim

Kn = K_nad/nt
d6 = kgs/(oaas*Keq)
dr1 = K_nad/nt
dr2 = psim*a_m
Pis = 2.4e-3
batp = katp/k1
dc = frt*psim
datp = b_m*at

e1 = pyrs/accs
e2 = pyrs/cits
e3 = pyrs/kgs
e4 = pyrs/oaas
e5 = pyrs/nt
e6 = pyrs/at
e7 = pyrs/psis/C_mito

to_psi = lambda dpsi : ((dpsi * 0.15) - PSI_min) / (PSI_max - PSI_min)
to_dpsi = lambda psi : ((psi*(PSI_max-PSI_min)) + PSI_min) / 0.15


class Mito(object):
    def __init__(self, pyr=1, acc=1, cit=1, akg=1, oaa=1,
                 nad=nads/nt, atp=0.8*atps/at, dpsi=psim,
                 atp_range=1000, baseline_atp=5, k_r=kr):
        self.pyr = pyr
        self.acc = acc
        self.cit = cit
        self.akg = akg
        self.oaa = oaa
        self.nad = nad
        self.atp = atp
        self.atpc = atp  # at equilibrium
        self.dpsi = dpsi
        self.Fkant = 0
        self.dFkantdt = 0
        self.atp_range = atp_range  # K_ant from outside is 1/ks
        self.baseline_atp = baseline_atp
        self.k_r = k_r
        self.psi = to_psi(self.dpsi)
        self.reaction_rates(current_atp=baseline_atp, current_leak=kl)
        self.derivatives()
        
    def reaction_rates(self, current_atp, current_leak):
        self.k_ant = current_atp / (self.atp_range)
        self.v1 = 1
        self.v2 = beta2*self.nad*self.pyr
        self.v3 = beta3*self.acc*self.oaa
        self.v4 = beta4*self.cit*self.nad
        self.v5 = beta5*self.akg*(1-self.atp)*self.nad
        self.v6 = beta6*(self.oaa-d6*self.akg)
        self.v7 = beta7*self.pyr*self.atp
        self.v8 = beta8*self.oaa
        self.vl = betal(current_leak)*self.dpsi
        # self.vant = bant(self.k_ant)*self.atp
        self.vant = (1+self.dFkantdt)*bant(self.k_ant)*self.atp
        # self.vant = (10**self.Fkant)*bant(self.k_ant)*self.atp
        self.vr = betar(self.k_r)*(1-self.nad)/(dr1+1-self.nad)/(1+exp(dr2*(self.dpsi-1)))
        acrit = Pis/(Pis+exp(dg0-dc*self.dpsi))
        self.vatp = batp*(2/(1+exp(datp*(self.atp-acrit)))-1)

    def fetch_actual_rates(self):
        ''' rates im uM/s '''
        print('Rates in uM/s, v_atp, v_ant, v_resp, v_leak')
        return self.vatp*k1/1e-6, self.vant*k1/1e-6, self.vr*k1/1e-6, self.vl*k1/1e-6

    def fetch_rate_const(self):
        ''' rates im uM/s '''
        print('Rates in uM/s, k_leak')
        return self.vl/(self.dpsi*1e-6)

    def fetch_actual_conc(self):
        '''concentrations in mM'''
        print('Concentration in mM, Pyr, NAD, ATP')
        return self.pyr*pyrs*1e3, self.nad*nt*1e3, self.atp*at*1e3

    def fetch_actual_psi(self):
        ''' psi at the inter membrane in mV'''
        print('PSI in mV')
        return self.dpsi*psim*1000
    
    def derivatives(self):
        self.dpyrdt = self.v1 - self.v2 - self.v7
        self.daccdt = (self.v2 - self.v3)*e1
        self.dcitdt = (self.v3 - self.v4)*e2
        self.dakgdt = (self.v4 + self.v6 - self.v5)*e3
        self.doaadt = (self.v5 + self.v7 - self.v3 - self.v8 - self.v6)*e4
        self.dnaddt = (-self.v2 - self.v4 - 2*self.v5 + self.vr)*e5
        self.datpdt = (self.vatp - self.vant + self.v5 - self.v7)*e6
        self.ddpsidt = (10*self.vr - self.vl - 3*self.vatp - self.vant)*e7
        self.dFkantdt = erf(10*(self.atp - self.atpc))/0.05
        
    def update_vals(self, dt, atp_cost=0, leak_cost=0):
        taum = dtm*dt  # scaling
        self.atpc = atp_cost
        # print(self.dFkantdt, self.Fkant, self.atpc)
        current_atp = self.baseline_atp  # + atp_cost
        current_leak = kl + leak_cost  # as a factor of kl
        self.reaction_rates(current_atp, current_leak)
        self.derivatives()
        self.pyr += self.dpyrdt*taum
        self.acc += self.daccdt*taum
        self.cit += self.dcitdt*taum
        self.akg += self.dakgdt*taum
        self.oaa += self.doaadt*taum
        self.nad += self.dnaddt*taum
        self.atp += self.datpdt*taum
        self.dpsi += self.ddpsidt*taum
        self.psi = to_psi(self.dpsi)
        self.Fkant += self.dFkantdt*taum
        
        
    def steadystate_vals(self, time=2000):
        print('Baseline ATP = ', self.baseline_atp)
        print('ATP Op.Range = ', self.atp_range)
        print('K ANT = ', self.baseline_atp / (self.atp_range))
        dt = 0.01
        for tt in np.arange(0, time, dt):
            self.update_vals(dt, atp_cost=self.atp, leak_cost=0)


if __name__ == '__main__':
    m = Mito(baseline_atp=53.87)
    m.steadystate_vals()
    print(m.fetch_actual_psi())
    print(m.fetch_actual_conc())
    print(m.fetch_actual_rates())
    print(m.fetch_rate_const())
