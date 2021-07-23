'''
Code for C. Nazaret et. al. / Journal of theoretical biology 2009
Original code in scilab (author Nazaret)
Ported to python3 (author chaitanya)

This code generates figure 7 and figure 8 from the paper.

'''

from math import exp
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def tca_etc(yy, tt):
    pyr, acc, cit, akg, oaa, nad, atp, dpsi = yy

    v1 = 1
    v2 = beta2*nad*pyr
    v3 = beta3*acc*oaa
    v4 = beta4*cit*nad
    v5 = beta5*akg*(1-atp)*nad
    v6 = beta6*(oaa-d6*akg)
    v7 = beta7*pyr*atp
    v8 = beta8*oaa
    vl = betal*dpsi
    vant = bant*atp
    vr = betar*(1-nad)/(dr1+1-nad)/(1+exp(dr2*(dpsi-1)))
    acrit = Pis/(Pis+exp(dg0-dc*dpsi))
    vatp = batp*(2/(1+exp(datp*(atp-acrit)))-1)

    dpyrdt = v1-v2-v7
    daccdt = (v2-v3)*e1
    dcitdt = (v3-v4)*e2
    dakgdt = (v4+v6-v5)*e3
    doaadt = (v5+v7-v3-v8-v6)*e4
    dnaddtul = (-v2-v4-2*v5+vr)*e5
    datpdt = (vatp-vant+v5-v7)*e6
    ddpsidt = (10*vr-vl-3*vatp-vant)*e7
    return [dpyrdt, daccdt, dcitdt, dakgdt, doaadt, dnaddt, datpdt, ddpsidt]

R = 8.314
T = 298
F = 96465

a = 100
b = 4
at = 4.16e-3
dg0 = 30540/R/T
frt = 1.2*3*F/R/T
h2opi = 1/2.4e-3

nt = 1.07e-3
asp = 1.6e-3
glu = 5.3e-3

psim = 0.150
C = 6.75e-3
K = 0.002
Keq = 0.12*glu/asp

# k1=0.000038
# k2=192.50253
# k3=72380.952
# k4=67.375887
# k5=57957.752
# k6=  0.0203014    
# k7= 28.011204
# k8=2.5333333
# kl= 0.0004267

kant = 0.0538700
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

beta2 = k2/k1*nt*pyrs
beta3 = k3/k1*accs*oaas
beta4 = k4/k1*cits*nt
beta5 = k5/k1*kgs*nt*at
beta6 = k6/k1*oaas
beta7 = k7/k1*pyrs*at
beta8 = k8/k1*oaas
betar = kr/k1
bant = kant/k1*at
betal = kl/k1*psim

Kn = K/nt
d6 = kgs/(oaas*Keq)
dr1 = K/nt
dr2 = psim*a

Pis = 2.4e-3
batp = katp/k1
dc = frt*psim
datp = b*at

e1 = pyrs/accs
e2 = pyrs/cits
e3 = pyrs/kgs
e4 = pyrs/oaas
e5 = pyrs/nt
e6 = pyrs/at
e7 = pyrs/psis/C

# Figure 8
x0 = [1, 1, 1, 1, 1, nads/nt, 1, 1] # High ATP state

# Figure 7
# x0 = [1, 1, 1, 1, 1, nads/nt, 0.8*atps/at, 1]  # Low ATP state

labels = ['pyr', 'acc', 'cit', 'akg', 'oaa', 'nad', 'atp', 'dpsi']
t = np.linspace(0, 50, num=1000)
y = odeint(tca_etc, x0, t)
for ii in range(8):
    plt.plot(t, y[:,ii], label=labels[ii])
plt.legend()
plt.show()
