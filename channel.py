from utils import AttributeDict


class Channel(object):
    def __init__(self, name, Gmax, E_r, gates=[], sp_gates=[]):
        self.name = name
        self.Gmax = Gmax
        self.E_r = E_r
        self.gates = gates
        self.gate_vals = AttributeDict()
        self.sp_gates = sp_gates
        self.I = 0

    def update_curr(self, dt, v, s=None):
        self.update_gates(dt, v, s)
        factor = 1
        for gate in self.gates:
            factor *= gate.get_val()**gate.count
        self.I = self.Gmax*factor*(v-self.E_r)
        return

    def update_rev(self, val):
        self.E_r = val

    def update_gates(self, dt, v, s):
        for g in self.gates:
            if g.name in self.sp_gates:
                vv = g.update_vals(dt, v, s)
            else:
                vv = g.update_vals(dt, v)
            self.gate_vals[g.name] = vv
        return

    def init_channel(self, v, s=None):
        for g in self.gates:
            if g.name in self.sp_gates:
                vv = g.init_val(v, s)
            else:
                vv = g.init_val(v)
            self.gate_vals[g.name] = vv
                

class ChannelKA(Channel):
    '''ac*((1-f)*b + f)'''
    def update_curr(self, dt, v, s=None):
        self.update_gates(dt, v, s)
        factor = 1
        for gate in [self.gates[0], self.gates[2]]:
            factor *= gate.get_val()**gate.count
        b = self.gates[1].get_val()**self.gates[1].count
        f = self.gates[3].get_val()  # frac that has no N-type
        self.I = self.Gmax*factor*((1-f)*b + f)*(v-self.E_r)
        return
