import numpy as np
from math import exp

# -------------------------------------
# K ANT =  0.001; PSI_max = 0.1865
# K ANT =  0.005; PSI_max = 0.1833

# K ANT = 0.2; PSI_min = 0.132
# K ANT =  0.5; PSI_min = 0.1288
# K ANT = 1; PSI_min = 0.1273
# -------------------------------------

PSI_max = 0.190
PSI_min = 0.125

R=8.314
T=298
F=96465

a_m = 100
b_m = 4
at = 4.16e-3
dg0 = 30540/R/T
frt = 1.2*3*F/R/T
h2opi = 1/2.4e-3

nt = 1.07e-3
asp = 1.6e-3
glu = 5.3e-3
psim = 0.150
C_mito = 6.75e-3
K_nad = 0.002
Keq = 0.12*glu/asp

# kant = 0.0538700 
katp = 0.1318967
kr = 0.0024905

k1 = 38e-6
k2 = 152
k3 = 57142
k4 = 53
k5 = 82361
k6 = 3.2e-3
k7 = 40
k8 = 3.6
kl = 0.426e-3

pyrs = 0.14e-3
accs = 0.07e-3
cits = 0.4e-3
kgs = 0.25e-3
oaas = 0.005e-3
nads = 0.94e-3
atps = 3.23e-3
psis = 0.150

dtm = k1 / pyrs

beta2 = k2/k1*nt*pyrs
beta3 = k3/k1*accs*oaas
beta4 = k4/k1*cits*nt
beta5 = k5/k1*kgs*nt*at
beta6 = k6/k1*oaas
beta7 = k7/k1*pyrs*at
beta8 = k8/k1*oaas
betar = lambda kR=kr : kR/k1
bant = lambda k_ant : k_ant/k1*at
betal = lambda kL=kl : kL/k1*psim

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

pyr_sig = lambda pyr : (1-(1/(1+np.exp(-10*(pyr-1.7)))))
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
        self.dpsi = dpsi
        self.atp_range = atp_range  # operational range of mito
        self.baseline_atp = baseline_atp
        self.k_r = k_r
        # self.psi = self.dpsi / (0.2 / 0.15)
        self.psi = to_psi(self.dpsi)
        self.reaction_rates(current_atp=baseline_atp, current_leak=kl)
        self.derivatives()
        
    def reaction_rates(self, current_atp, current_leak):
        self.k_ant = current_atp / (self.atp_range)
        # if self.k_ant > 0.1:
        #     pyr_corr = pyr_sig(self.pyr) 
        # else:
        #     pyr_corr = 1
        pyr_corr = 1
        self.v1 = 1 * pyr_corr
        self.v2 = beta2*self.nad*self.pyr * pyr_corr
        self.v3 = beta3*self.acc*self.oaa * pyr_corr
        self.v4 = beta4*self.cit*self.nad * pyr_corr
        self.v5 = beta5*self.akg*(1-self.atp)*self.nad * pyr_corr
        self.v6 = beta6*(self.oaa-d6*self.akg) * pyr_corr
        self.v7 = beta7*self.pyr*self.atp * pyr_corr
        self.v8 = beta8*self.oaa * pyr_corr
        self.vl = betal(current_leak)*self.dpsi * pyr_corr
        self.vant = bant(self.k_ant)*self.atp * pyr_corr
        self.vr = betar(self.k_r)*(1-self.nad)/(dr1+1-self.nad)/(1+exp(dr2*(self.dpsi-1))) * pyr_corr
        acrit = Pis/(Pis+exp(dg0-dc*self.dpsi))
        self.vatp = batp*(2/(1+exp(datp*(self.atp-acrit)))-1) * pyr_corr        

    def fetch_actual_rates(self):
        ''' rates im uM/s '''
        print('Rates in uM/s, v_atp, v_ant, v_resp')
        return self.vatp*k1/1e-6, self.vant*k1/1e-6, self.vr*k1/1e-6

    def fetch_actual_conc(self):
        '''concentrations in mM'''
        print('Concentration in uM/s, Pyr, NAD, ATP')
        return self.pyr*pyrs/1e3, self.nad*nt/1e3, self.atp*at/1e3

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
        self.ddpsidt = (10*self.vr- self.vl- 3*self.vatp - self.vant)*e7

    def update_vals(self, dt, atp_cost=0, leak_cost=0):
        taum = dtm*dt  # scaling
        current_atp = self.baseline_atp + atp_cost
        current_leak = kl + leak_cost # as a factor of kl instead of normalizing
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
        # self.psi = self.dpsi / (0.2 / 0.15)  # re-normed version handy
        self.psi = to_psi(self.dpsi)

    def clamped_vals(self, dt, atp, dpsi):
        # This approach here is wrong
        taum = dtm*dt  # scaling
        self.reaction_rates(self.baseline_atp, kl)
        self.derivatives()
        self.pyr += self.dpyrdt*taum
        self.acc += self.daccdt*taum
        self.cit += self.dcitdt*taum
        self.akg += self.dakgdt*taum
        self.oaa += self.doaadt*taum
        self.nad += self.dnaddt*taum
        self.atp = atp # self.datpdt*taum
        self.dpsi = dpsi # self.ddpsidt*taum
        # self.psi = self.dpsi / (0.2 / 0.15)  # re-normed version handy
        self.psi = to_psi(self.dpsi)
        
    def steadystate_vals(self, time=2000):
        print('Baseline ATP = ', self.baseline_atp)
        print('ATP Op.Range = ', self.atp_range)
        print('K ANT = ', self.baseline_atp / (self.atp_range))
        dt = 0.01
        for tt in np.arange(0, time, dt):
            self.update_vals(dt, atp_cost=0, leak_cost=0)

if __name__ == '__main__':
    m = Mito(baseline_atp=200)
    m.steadystate_vals()
    print(m.fetch_actual_psi())
    print(m.fetch_actual_conc())
    print(m.fetch_actual_rates())
    # print(m.atp*at, m.dpsi*psim, m.pyr*pyrs)
    # print(m.fetch_rates())
