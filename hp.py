#!./env/bin/python3

import argparse
from sys import stdin
from scipy.optimize import curve_fit
from scipy.integrate import ode
from scipy.interpolate import interp1d
from numpy import ndarray, array
from functools import reduce


default_mode = "temp"
sig_kw = "known"
resp_kw = "unknown"
val_delim = ' '
outside_temp = 20 # Celcius


def main():
    args = parser.parse_args()
    mode = get_mode(args)
    # print(mode)
    known_signal = get_known_signal()
    known_response = get_known_response(mode)
    emit_predicted_response(known_signal, known_response, mode)





parser = argparse.ArgumentParser(
            description="heat prediction script to calibrate faster"
         )
parser.add_argument("mode", type = str, nargs = '?',
                    help = "\"temp\" or \"pow\" to "
                           "predict by known heater temperatures "
                           "or by known heater power")

def get_mode(args):
    mode = args.mode
    try:
        if mode is None:
            # TODO: verbose flag
            print(f"mode defaults to {default_mode}")
            return default_mode
        while not ( "temp" == mode or "pow" == mode ):
            mode = input(f"specified mode is \"{mode}\", which is neither \"temp\" nor \"pow\"\n"
                          "type one of those and press Enter\n")
    except EOFError:
        print("\ncouldn't get mode --- exiting...")
        exit(1)
    except KeyboardInterrupt:
        print("\nexiting...")
        exit(0)
    return mode

def get_known_signal():
    try:
        for line in stdin:
            if sig_kw == line.rstrip():
                break
        known_signal = dict()
        for line in stdin:
            stripped = line.rstrip()
            if "" == stripped:
                break
            if resp_kw == stripped:
                print(f"separate {sig_kw} and {resp_kw} sections with empty line")
                print("exiting...")
                exit(2)
            try:
                sig_t, sig = map(float, stripped.split(val_delim))
                known_signal[sig_t] = sig
            except:
                print(f"discarded \"{stripped}\"")
        return known_signal
    except EOFError:
        print("exiting on unexpected end of file...")
        exit(3)
    
# def floater(string):
#     try:
#         return float(string)
#     except:
#         return None
def get_known_response(mode):
    try:
        for line in stdin:
            if resp_kw == line.rstrip():
                break
        known_response = dict()
        for line in stdin:
            stripped = line.rstrip()
            try:
                if "temp" == mode:
                    time, body_temp = map(float, stripped.split(val_delim))
                    known_response[time] = body_temp
                elif "pow" == mode:
                    time, body_temp, heater_temp = map(float, stripped.split(val_delim))
                    if time is None:
                        raise IOError("known signal time not specified")
                    known_response[time] = (body_temp, heater_temp)
                else:
                    print(f"unexpected mode \"{mode}\", probably bug...")
                    print("exiting...")
                    exit(4)
            except:
                print(f"discarded \"{stripped}\"")
    except EOFError:
        print("exiting on unexpected end of file")
        exit(5)
    return known_response


def emit_predicted_response(known_signal, known_response, mode):
    known_fn = PseudoFunction(known_signal)
    unknown_fn = PseudoFunction(known_response)
    # print(f"known:\n{known_fn.table}", end='')
    # print(f"unknown:\n{unknown_fn.table}", end='')
    predictor = Predictor(known_fn, unknown_fn, mode)
    predictor.emit_all()


class PseudoFunction:
    def __init__(self, table_dict):
        self.table = Table.FromDict(table_dict)
        # WARNING: out of bounds not handled
        self.lboundary = self.table.keys[0]
        self.rboundary = self.table.keys[-1]
        self.interpolator = interp1d(self.table.keys, self.table.vals,
                                     fill_value="extrapolate", bounds_error=False,
                                     assume_sorted=True)

    def __call__(self, arg):
        # WARNING: return type is array, not float
        # print(f"table:\n{self.table}", end='')
        # print(f"is {arg} in {self.lboundary} --- {self.rboundary}")
        return self.interpolator(arg)
        # return float(self.interpolator(arg))


def eq(l, r):
    return abs(l - r) < 1e-07
class Table:
    def __init__(self, keys, vals):
        self.keys = keys
        self.vals = vals

    @classmethod
    def FromDict(cls, _dict):
        keys = []
        vals = []
        for k in sorted(_dict):
            keys.append(k)
            vals.append(_dict[k])
        return cls(keys, vals)

    def submerged_keys(self, other):
        # self sets result's boundaries
        # [2, 4], [1, 3, 5] -> [2, 3, 4]
        res: list[float] = []
        s, o = 0, 0
        while other.keys[o] < self.keys[s]:
            o += 1
        while s < len(self.keys) and o < len(other.keys):
            if eq(self.keys[s], other.keys[o]):
                res.append(self.keys[s])
                s += 1
                o += 1
            elif other.keys[o] < self.keys[s]:
                res.append(other.keys[o])
                o += 1
            else:
                res.append(self.keys[s])
                s += 1
        while s < len(self.keys):
            res.append(self.keys[s])
            s += 1
        return res

    def __str__(self):
        res = ""
        for i in range(len(self.keys)):
            res += f"{self.keys[i]} -> {self.vals[i]}\n"
        return res


class Predictor:
    def __init__(self, known_fn: PseudoFunction, unknown_fn: PseudoFunction, mode):
        self.kf = known_fn
        self.uf = unknown_fn
        self.mode = mode
        if "temp" == mode:
            def _rhs(time, body_temp, heater_temp, kappa, kappa0):
                # heater_temp = self.kf(time)
                return kappa*(heater_temp - body_temp) + \
                       kappa0*(outside_temp - body_temp)
            def _jac(time, body_temp, heater_temp, kappa, kappa0):
                return -kappa - kappa0
        elif "pow" == mode:
            pass
        else:
            print(f"unrecognized mode \"{mode}\", probably bug, exiting...")
            exit(6)
        def rhs(t, x, *params):
            # print("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        # print(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        self.integrator = Integrator(rhs, jac, known_fn(t0), self.times)
        guess = self.generate_guess()
        self.params, _ = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess)
        self.params = tuple(self.params)

    def generate_guess(self) -> tuple:
        if "temp" == self.mode:
            k0 = 0
            ks: list[float] = []
            for i in range(min(10, len(self.uf.table.keys)-1)):
                # body temperatures
                bt_i00 = self.uf.table.vals[i]
                bt_i01 = self.uf.table.vals[i+1]
                bt_i05 = 0.5 * (bt_i00 + bt_i01)
                # heater temperatures
                ht_i00 = self.kf(bt_i00)
                ht_i01 = self.kf(bt_i01)
                ht_i05 = 0.5 * (ht_i00 + ht_i01)
                dt = self.uf.table.keys[i+1] - self.uf.table.keys[i]
                ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
            return (k, k0)
        else: # "pow"
            pass

    def prepare_to_emit(self, *params) -> tuple[ndarray, ndarray]:
        ts: list[float] = []
        vals: list[ndarray] = []
        # print(f"preparing to emit for times {self.times}")
        for t in self.times:
            ts.append(t)
            vals.append(self.integrator(t, *params))
        return (array(ts), array(vals))

    def emit_all(self):
        ts, vals = self.prepare_to_emit(*self.params)
        for i in range(len(ts)):
            v = vals[i]
            try:
                v[0]
            except IndexError:
                v = [v]
            # print(f"printing {ts[i]} and {vals[i]}")
            print(ts[i], *tuple(v))



class Integrator:
    def __init__(self, rhs, jac, ic, times):
        self.rhs = rhs
        self.jac = jac
        self.ic = ic
        self.times = times
        self.pf = PseudoFunction({0:0, 1:0})
        self.params = tuple()

    def __call__(self, arg, *params):
        # print(f"integrator call at {arg}, with {params}")
        self.set_params(*params)
        return self.pf(arg)

    def set_params(self, *params):
        if self.params != params:
            # print(f"resetting integrator parameters to {params}")
            self.params = params
            self.integrate()

    def integrate(self):
        solver = ode(lambda t, x: self.rhs(t, x, *self.params),
                     lambda t, x: self.jac(t, x, *self.params))
        # solver = ode(self.rhs, self.jac)
        time2val = dict()
        # WARNING: out of bounds not handled
        t0, tend = self.times[0], self.times[-1]
        time2val[t0] = self.ic
        solver.set_initial_value(self.ic, t0)
        # print("setting solver parameters to", *self.params)
        # solver.set_f_params(*self.params).set_jac_params(*self.params)
        for time in self.times[1::]:
            if solver.successful():
                time2val[time] = solver.integrate(time)
            else:
                break
        self.pf = PseudoFunction(time2val)


if __name__ == "__main__":
    main()

