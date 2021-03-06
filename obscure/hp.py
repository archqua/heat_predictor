#!/bin/env python

import argparse
from sys import stdin, stderr
from scipy.optimize import curve_fit, basinhopping
import scipy.optimize as so
from scipy.integrate import ode
from scipy.interpolate import interp1d
import numpy as np
from numpy import inf as ninf
from numpy import ndarray, array
from functools import reduce


default_mode = "temp"
sig_kw = "known"
resp_kw = "unknown"
val_delim = ','
outside_temp = 20 # Celcius

# valid_modes = {"temp", "temp_offset",
#                "2isol", "2isol_offset",
#                "stiff_offset", "stiff_square_offset",
#                "stiff_basin_offset"
#                }
temp_modes = {"temp", "temp_offset",
              "2isol", "2isol_offset",
              "stiff_offset", "stiff_square_offset",
              "stiff_basin_offset",
              "stiff_penalty_offset",
              }
pow_modes = {"pow"}
valid_modes = temp_modes.union(pow_modes)

def notify(*args, **kwargs):
    kwargs['file'] = stderr
    print(*args, **kwargs)


def main():
    args = parser.parse_args()
    # notify(args.mode, args.skip)
    mode = get_mode(args)
    known_signal = get_known_signal(args.skip)
    known_response = get_known_response(mode, args.skip)
    emit_predicted_response(known_signal, known_response, mode)





parser = argparse.ArgumentParser(
            description="heat prediction script to calibrate faster"
         )
parser.add_argument("--mode", "-m", type = str, #nargs = 1,
                    help = f"one of {valid_modes}")
parser.add_argument("--skip", "-s", type = int, #nargs = 1,
                    default = 1,
                    help = "only each Nth line is not ignored",
                    dest="skip")

def get_mode(args):
    mode = args.mode
    try:
        if mode is None:
            # TODO: verbose flag
            notify(f"mode defaults to {default_mode}", file=stderr)
            return default_mode
        while not ( mode in valid_modes ):
            mode = input(f"specified mode is \"{mode}\", which is neither of {valid_modes}\n"
                          "type one of those and press Enter\n")
    except EOFError:
        notify("\ncouldn't get mode --- exiting...", file=stderr)
        exit(1)
    except KeyboardInterrupt:
        notify("\nexiting...", file=stderr)
        exit(1)
    return mode

def get_known_signal(skip):
    try:
        for line in stdin:
            if sig_kw == line.rstrip():
                break
        known_signal = dict()
        i = 0
        for line in stdin:
            stripped = line.rstrip()
            if "" == stripped:
                break
            if resp_kw == stripped:
                notify(f"separate {sig_kw} and {resp_kw} sections with empty line", file=stderr)
                notify("exiting...", file=stderr)
                exit(2)
            try:
                sig_t, sig = map(float, stripped.split(val_delim))
                if (i % skip == 0):
                    known_signal[sig_t] = sig
                i += 1
            except:
                notify(f"discarded \"{stripped}\"", file=stderr)
        return known_signal
    except EOFError:
        notify("exiting on unexpected end of file...", file=stderr)
        exit(3)
    
# def floater(string):
#     try:
#         return float(string)
#     except:
#         return None
def get_known_response(mode, skip):
    try:
        for line in stdin:
            if resp_kw == line.rstrip():
                break
        known_response = dict()
        i = 0
        for line in stdin:
            stripped = line.rstrip()
            try:
                if mode in temp_modes:
                    time, body_temp = map(float, stripped.split(val_delim))
                    if (i % skip == 0):
                        known_response[time] = body_temp
                elif mode in pow_modes:
                    time, body_temp, heater_temp = map(float, stripped.split(val_delim))
                    if time is None:
                        raise IOError("known signal time not specified")
                    if (i % skip == 0):
                        known_response[time] = (body_temp, heater_temp)
                else:
                    notify(f"unexpected mode \"{mode}\", probably bug...", file=stderr)
                    notify("exiting...", file=stderr)
                    exit(4)
                i += 1
            except:
                notify(f"discarded \"{stripped}\"", file=stderr)
    except EOFError:
        notify("exiting on unexpected end of file", file=stderr)
        exit(5)
    return known_response


def emit_predicted_response(known_signal, known_response, mode):
    known_fn = PseudoFunction(known_signal)
    unknown_fn = PseudoFunction(known_response)
    predictor = Predictors.get_predictor(known_fn, unknown_fn, mode)
    if predictor is None:
        notify(f"unknown mode {mode}", file=stderr)
        notify("exiting...", file=stderr)
        exit(6)
    params = predictor.dictonize()
    notify(params, file=stderr)
    predictor.emit_all()


class PseudoFunction:
    def __init__(self, table_dict):
        self.table = Table.FromDict(table_dict)
        # WARNING: out of bounds not handled
        self.lboundary = self.table.keys[0]
        self.rboundary = self.table.keys[-1]
        # notify(f"constructing pf from\n{self.table}", file=stderr)
        self.interpolator = interp1d(self.table.keys, self.table.vals,
                                     fill_value="extrapolate", bounds_error=False,
                                     assume_sorted=True,
                                     axis=0,
                                     )

    def __call__(self, arg):
        # WARNING: return type is array, not float
        # notify(f"table:\n{self.table}", end='')
        # notify(f"is {arg} in {self.lboundary} --- {self.rboundary}")
        return self.interpolator(arg)
        # return float(self.interpolator(arg))


def eq(l, r):
    return abs(l - r) < 1e-07
class Table:
    def __init__(self, keys, vals):
        self.keys = np.array(keys)
        self.vals = np.array(vals)

    @classmethod
    def FromDict(cls, _dict):
        keys = []
        vals = []
        for k in sorted(_dict):
            keys.append(np.array(k))
            vals.append(np.array(_dict[k]))
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
        return np.array(res)

    def __str__(self):
        res = ""
        for i in range(len(self.keys)):
            res += f"{self.keys[i]} -> {self.vals[i]}\n"
        return res


class Integrator:
    def __init__(self, rhs, jac, times, ic=None):
        self.rhs = rhs
        self.jac = jac
        self.ic = ic
        self.times = times
        self.pf = PseudoFunction({0:0, 1:0})
        self.params = tuple()

    def __call__(self, arg, *params):
        # notify(f"integrator call at {arg}, with {params}")
        self.set_params(*params)
        return self.pf(arg)

    def set_params(self, *params):
        if self.params != params[1:] or self.ic != params[0]:
            # notify(f"resetting integrator parameters to {params}")
            self.ic = params[0]
            self.params = params[1:]
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
        # notify("setting solver parameters to", *self.params, file=stderr)
        # solver.set_f_params(*self.params).set_jac_params(*self.params)
        for time in self.times[1::]:
            if solver.successful():
                time2val[time] = solver.integrate(time)
            else:
                break
        self.pf = PseudoFunction(time2val)


class IPredictor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"AbstractPredictor: __init__({args}, {kwargs})")

    def emit_all(self):
        raise NotImplementedError("AbstractPredictor: emit_all()")

    def set(self, *args, **kwargs):
        raise NotImplementedError(f"AbstractPredictor: set({args, kwargs})")

    def dictonize(self):
        raise NotImplementedError("AbstractPredictor: dictonize()")

    def connect(self, successor):
        self.successor = successor
        return self

    def find(self, mode):
        if self.mode == mode:
            return self
        else:
            return self.successor.find(mode)

    def null_init(self, mode, known_fn, unknown_fn, successor):
        self.mode = mode
        self.successor = successor
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = None
        self.integrator = None
        self.params = None
        self.is_set = False


class TempPredictor(IPredictor):
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("temp", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, body_temp, heater_temp, kappa, kappa0):
            # heater_temp = self.kf(time)
            return kappa*(heater_temp - body_temp) + \
                   kappa0*(outside_temp - body_temp)
        def _jac(time, body_temp, heater_temp, kappa, kappa0):
            return -kappa - kappa0
        bounds = (array([self.uf(t0)*0.9, 0, 0]), 
                  array([self.uf(t0)*1.1, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        self.integrator = Integrator(rhs, jac, self.times, unknown_fn(t0))
        guess = self.generate_guess()
        self.params, _ = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        self.params = tuple(self.params)
        return self

    def generate_guess(self) -> tuple:
        if self.kf is not None and self.uf is not None and self.times is not None:
            k0 = 0
            ks: list[float] = []
            for tl, tr in zip(self.times[:min(9, len(self.times) - 1)],
                              self.times[1:min(10, len(self.times))]):
                # body temperatures
                bt_i00 = self.uf(tl)
                bt_i01 = self.uf(tr)
                bt_i05 = 0.5 * (bt_i00 + bt_i01)
                # heater temperatures
                ht_i00 = self.kf(tl)
                ht_i01 = self.kf(tr)
                ht_i05 = 0.5 * (ht_i00 + ht_i01)
                dt = tr - tl
                ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            # for i in range(min(10, len(self.uf.table.keys)-1)):
            #     # body temperatures
            #     bt_i00 = self.uf.table.vals[i]
            #     bt_i01 = self.uf.table.vals[i+1]
            #     bt_i05 = 0.5 * (bt_i00 + bt_i01)
            #     # heater temperatures
            #     ht_i00 = self.kf(bt_i00)
            #     ht_i01 = self.kf(bt_i01)
            #     ht_i05 = 0.5 * (ht_i00 + ht_i01)
            #     dt = self.uf.table.keys[i+1] - self.uf.table.keys[i]
            #     ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
            return (self.uf(self.times[0]), k, k0)
        else:
            raise ValueError("TempPredictor: not set to generate guess")
        

    def prepare_to_emit(self, *params) -> tuple[ndarray, ndarray]:
        if self.times is not None and self.integrator is not None:
            ts: list[float] = []
            vals: list[ndarray] = []
            # notify(f"preparing to emit for times {self.times}")
            for t in self.times:
                ts.append(t)
                vals.append(self.integrator(t, *params))
            return (array(ts), array(vals))
        else:
            raise ValueError("TempPredictor: not set to prepare to emit")

    def emit_all(self):
        ts, vals = self.prepare_to_emit(*self.params)
        for i in range(len(ts)):
            v = vals[i]
            try:
                v[0]
            except IndexError:
                v = [v]
            # notify(f"printing {ts[i]} and {vals[i]}")
            print(ts[i], *tuple(v))

    def dictonize(self):
        res = dict()
        res['t0'] = self.params[0]
        res['k']  = self.params[1]
        res['k0'] = self.params[2]
        return res


class TempOffsetPredictor(IPredictor):
    def __init__(self, known_fn: PseudoFunction = None, unknown_fn: PseudoFunction = None, successor = None):
        self.null_init("temp_offset", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, body_temp, heater_temp, kappa, kappa0, toffset):
            # heater_temp = self.kf(time)
            return kappa*(heater_temp + toffset - body_temp) + \
                   kappa0*(outside_temp - body_temp)
        def _jac(time, body_temp, heater_temp, kappa, kappa0, toffset):
            return -kappa - kappa0
        bounds = (array([self.uf(t0)*0.9, 0,    0,   -ninf]), 
                  array([self.uf(t0)*1.1, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        self.integrator = Integrator(rhs, jac, self.times, unknown_fn(t0))
        guess = self.generate_guess()
        self.params, _ = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        self.params = tuple(self.params)
        return self

    def generate_guess(self) -> tuple:
        if self.kf is not None and self.uf is not None and self.times is not None:
            k0 = 0
            toffset = 0
            ks: list[float] = []
            for tl, tr in zip(self.times[:min(9, len(self.times) - 1)],
                              self.times[1:min(10, len(self.times))]):
                # body temperatures
                bt_i00 = self.uf(tl)
                bt_i01 = self.uf(tr)
                bt_i05 = 0.5 * (bt_i00 + bt_i01)
                # heater temperatures
                ht_i00 = self.kf(tl)
                ht_i01 = self.kf(tr)
                ht_i05 = 0.5 * (ht_i00 + ht_i01)
                dt = tr - tl
                ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
            return (self.uf(self.times[0]), k, k0, toffset)
        else:
            raise ValueError("TempOffsetPredictor: not set to prepare to emit")

    def prepare_to_emit(self, *params) -> tuple[ndarray, ndarray]:
        if self.times is not None and self.integrator is not None:
            ts: list[float] = []
            vals: list[ndarray] = []
            # notify(f"preparing to emit for times {self.times}")
            for t in self.times:
                ts.append(t)
                vals.append(self.integrator(t, *params))
            return (array(ts), array(vals))
        else:
            raise ValueError("TempOffsetPredictor: not set to prepare to emit")

    def emit_all(self):
        ts, vals = self.prepare_to_emit(*self.params)
        for i in range(len(ts)):
            v = vals[i]
            try:
                v[0]
            except IndexError:
                v = [v]
            # notify(f"printing {ts[i]} and {vals[i]}")
            print(ts[i], *tuple(v))

    def dictonize(self):
        res = dict()
        res['t0']      = self.params[0]
        res['k']       = self.params[1]
        res['k0']      = self.params[2]
        res['toffset'] = self.params[3]
        return res


class TwoSideIsolated(IPredictor):
    """ heater <-> inner <-> body <-> outer
                   outer            inner,out """
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("2isol", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, ts, heater_temp, k01, k02, k1, k2, k12):
            """ body_temp = ts[0]
                inner_temp = ts[1]
                outer_temp = ts[2] """
            # heater_temp = self.kf(time)
            return array((k01*(ts[1] - ts[0])       + k02*(ts[2] - ts[0]),
                          k1*(heater_temp - ts[1])  + k01*(ts[0] - ts[1]) + k12*(ts[2] - ts[1]),
                          k2*(outside_temp - ts[2]) + k02*(ts[0] - ts[2]) + k12*(ts[1] - ts[2]),
                          ))
        def _jac(time, ts, heater_temp, k01, k02, k1, k2, k12):
            return array([[-k01 - k02, k01,            k02],
                          [k01,       -k01 - k1 - k12, k12],
                          [k02,        k12,           -k02 - k2 - k12]
                          ])
        bounds = (array([self.uf(t0)*0.9, self.uf(t0)*0.9, outside_temp,    0,    0,    0,    0,    0]), 
                  array([self.uf(t0)*1.1, self.kf(t0)*1.1, self.uf(t0)*1.1, ninf, ninf, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        init_t0 = self.uf(t0)
        init_t1 = 0.5*(init_t0 + self.kf(t0))
        init_t2 = 0.5*(init_t0 + outside_temp)
        self.integrator = TwoSideIsolated.IntegrAdapter(rhs, jac, self.times,
                                                        array([init_t0, init_t1, init_t2,]),
                                                        )
        guess = self.generate_guess()
        self.params, _ = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        self.params = tuple(self.params)
        return self

    def generate_guess(self) -> tuple:
        if self.kf is not None and self.uf is not None and self.times is not None:
            k02 = 0
            k2 = 0
            k12 = 0
            toffset = 0
            ks: list[float] = []
            for tl, tr in zip(self.times[:min(9, len(self.times) - 1)],
                              self.times[1:min(10, len(self.times))]):
                # body temperatures
                bt_i00 = self.uf(tl)
                bt_i01 = self.uf(tr)
                bt_i05 = 0.5 * (bt_i00 + bt_i01)
                # heater temperatures
                ht_i00 = self.kf(tl)
                ht_i01 = self.kf(tr)
                ht_i05 = 0.5 * (ht_i00 + ht_i01)
                dt = tr - tl
                ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
            k01 = k/2
            k1 = k/2
            init_t0 = self.uf(self.times[0])
            init_t1 = 0.5*(init_t0 + self.uf(self.times[0]))
            init_t2 = 0.5*(init_t0 + outside_temp)
            return (init_t0, init_t1, init_t2, k01, k02, k1, k2, k12)
        else:
            raise ValueError("TwoSideIsolated: not set to prepare to emit")

    def prepare_to_emit(self, *params) -> tuple[ndarray, ndarray]:
        if self.times is not None and self.integrator is not None:
            ts: list[float] = []
            vals: list[ndarray] = []
            # notify(f"preparing to emit for times {self.times}")
            for t in self.times:
                ts.append(t)
                vals.append(self.integrator(t, *params))
            return (array(ts), array(vals))
        else:
            raise ValueError("TwoSideIsolated: not set to prepare to emit")

    def emit_all(self):
        ts, vals = self.prepare_to_emit(*self.params)
        for i in range(len(ts)):
            v = vals[i]
            try:
                v[0]
            except IndexError:
                v = [v]
            # notify(f"printing {ts[i]} and {vals[i]}")
            print(ts[i], *tuple(v))

    class IntegrAdapter(Integrator):
        def set_params(self, init_t0, init_t1, init_t2, k01, k02, k1, k2, k12):
            ic = array([init_t0, init_t1, init_t2])
            params = (k01, k02, k1, k2, k12)
            if (self.ic != ic).any() or self.params != params:
                self.ic = ic
                self.params = params
                self.integrate()

        def __call__(self, arg, init_t0, init_t1, init_t2, *params):
            self.set_params(init_t0, init_t1, init_t2, *params)
            # notify(f"IntegrAdapter: call({arg}, {(init_t0, init_t1, init_t2)}, {params}) -> {self.pf(arg)} -> {self.pf(arg)[0]}", file=stderr)
            try:
                return self.pf(arg)[:, 0]
            except IndexError:
                return self.pf(arg)[0]
                pass
        
    def dictonize(self):
        res = dict()
        for k, v in zip(['t0', 't1', 't2', 'k01', 'k02', 'k1', 'k2', 'k12'], self.params):
            res[k] = v
        return res


class TwoSideIsolatedOffset(IPredictor):
    """ heater <-> inner <-> body <-> outer
                   outer            inner,out """
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("2isol_offset", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, ts, heater_temp, k01, k02, k1, k2, k12, toffset):
            """ body_temp  = ts[0]
                inner_temp = ts[1]
                outer_temp = ts[2] """
            # heater_temp = self.kf(time)
            return array((k01*(ts[1] - ts[0])       + k02*(ts[2] - ts[0]),
                          k1*(heater_temp + toffset - ts[1])  + k01*(ts[0] - ts[1]) + k12*(ts[2] - ts[1]),
                          k2*(outside_temp - ts[2]) + k02*(ts[0] - ts[2]) + k12*(ts[1] - ts[2]),
                          ))
        def _jac(time, ts, heater_temp, k01, k02, k1, k2, k12, toffset):
            return array([[-k01 - k02, k01,            k02],
                          [k01,       -k01 - k1 - k12, k12],
                          [k02,        k12,           -k02 - k2 - k12]
                          ])
        bounds = (array([self.uf(t0)*0.9, self.uf(t0)*0.9, outside_temp,    0,    0,    0,    0,    0,   -ninf]), 
                  array([self.uf(t0)*1.1, self.kf(t0)*1.1, self.uf(t0)*1.1, ninf, ninf, ninf, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        init_t0 = self.uf(t0)
        init_t1 = 0.5*(init_t0 + self.kf(t0))
        init_t2 = 0.5*(init_t0 + outside_temp)
        self.integrator = TwoSideIsolatedOffset.IntegrAdapter(
                                                    rhs, jac, self.times,
                                                    array([init_t0, init_t1, init_t2,]),
                                                    )
        guess = self.generate_guess()
        self.params, pcov = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   # method='dogbox',
                                   # jac='3-step'
                                   )
        self.params = tuple(self.params)
        notify(f"sigma: {np.sqrt(np.diag(pcov))}", file=stderr)
        return self

    def generate_guess(self) -> tuple:
        if self.kf is not None and self.uf is not None and self.times is not None:
            k02 = 0
            k2 = 0
            k12 = 0
            toffset = 0
            ks: list[float] = []
            for tl, tr in zip(self.times[:min(9, len(self.times) - 1)],
                              self.times[1:min(10, len(self.times))]):
                # body temperatures
                bt_i00 = self.uf(tl)
                bt_i01 = self.uf(tr)
                bt_i05 = 0.5 * (bt_i00 + bt_i01)
                # heater temperatures
                ht_i00 = self.kf(tl)
                ht_i01 = self.kf(tr)
                ht_i05 = 0.5 * (ht_i00 + ht_i01)
                dt = tr - tl
                ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
            k01 = k/2
            k1 = k/2
            init_t0 = self.uf(self.times[0])
            init_t1 = 0.5*(init_t0 + self.uf(self.times[0]))
            init_t2 = 0.5*(init_t0 + outside_temp)
            return (init_t0, init_t1, init_t2, k01, k02, k1, k2, k12, toffset)
        else:
            raise ValueError("TwoSideIsolated: not set to prepare to emit")

    def prepare_to_emit(self, *params) -> tuple[ndarray, ndarray]:
        if self.times is not None and self.integrator is not None:
            ts: list[float] = []
            vals: list[ndarray] = []
            # notify(f"preparing to emit for times {self.times}")
            for t in self.times:
                ts.append(t)
                vals.append(self.integrator(t, *params))
            return (array(ts), array(vals))
        else:
            raise ValueError("TwoSideIsolated: not set to prepare to emit")

    def emit_all(self):
        ts, vals = self.prepare_to_emit(*self.params)
        for i in range(len(ts)):
            v = vals[i]
            try:
                v[0]
            except IndexError:
                v = [v]
            # notify(f"printing {ts[i]} and {vals[i]}")
            print(ts[i], *tuple(v))

    class IntegrAdapter(Integrator):
        def set_params(self, init_t0, init_t1, init_t2, k01, k02, k1, k2, k12, toffset):
            ic = array([init_t0, init_t1, init_t2])
            params = (k01, k02, k1, k2, k12, toffset)
            if (self.ic != ic).any() or self.params != params:
                self.ic = ic
                self.params = params
                self.integrate()

        def __call__(self, arg, init_t0, init_t1, init_t2, *params):
            self.set_params(init_t0, init_t1, init_t2, *params)
            # notify(f"IntegrAdapter: call({arg}, {(init_t0, init_t1, init_t2)}, {params}) -> {self.pf(arg)} -> {self.pf(arg)[0]}", file=stderr)
            try:
                return self.pf(arg)[:, 0]
            except IndexError:
                return self.pf(arg)[0]
                pass

    def dictonize(self):
        res = dict()
        for k, v in zip(['t0', 't1', 't2', 'k01', 'k02', 'k1', 'k2', 'k12', 'toffset'], self.params):
            res[k] = v
        return res


class StiffOffset(IPredictor):
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("stiff_offset", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            """ fast_temp = ts[0]
                slow_temp = ts[1]
                """
            # heater_temp = self.kf(time)
            return array((k_f*(heater_temp + toffset - ts[0]),
                          k_s*(heater_temp + toffset - ts[1]),
                          ))
        def _jac(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            return array([[-k_f, 0  ],
                          [ 0,  -k_s],
                          ])
        ############## alpha,   fast_temp,     slow_temp,    k_f,  k_s,  offset
        bounds = (array([0, self.uf(t0)*0.8, self.uf(t0)*0.8, 0,    0,   -ninf]), 
                  array([1, self.uf(t0)*1.2, self.uf(t0)*1.2, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        init_tf = self.uf(t0)
        init_ts = self.uf(t0)
        self.integrator = StiffOffset.IntegrAdapter(
                                                    rhs, jac, self.times,
                                                    array([init_tf, init_ts,]),
                                                    )
        guess = self.generate_guess()
        self.params, pcov = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   # method='dogbox',
                                   # jac='3-step'
                                   )
        self.params = tuple(self.params)
        notify(f"sigma: {np.sqrt(np.diag(pcov))}", file=stderr)
        return self

    class IntegrAdapter(Integrator):
        def set_params(self, alpha, init_tf, init_ts, k_f, k_s, toffset):
            ic = array([init_tf, init_ts])
            params = (alpha, k_f, k_s, toffset)
            if (self.ic != ic).any() or self.params != params:
                self.ic = ic
                self.params = params
                self.integrate()

        def __call__(self, arg, *params):
            self.set_params(*params)
            alpha = params[0]
            try:
                return alpha*self.pf(arg)[:, 0] + (1-alpha)*self.pf(arg)[:, 1]
            except IndexError:
                return alpha*self.pf(arg)[0] + (1-alpha)*self.pf(arg)[1]
                pass

    def generate_guess(self) -> tuple:
        if self.kf is not None and self.uf is not None and self.times is not None:
            # alpha = 0.85 # good one
            alpha = 0.8 # worse one
            toffset = 0
            ks: list[float] = []
            for tl, tr in zip(self.times[:min(9, len(self.times) - 1)],
                              self.times[1:min(10, len(self.times))]):
                # body temperatures
                bt_i00 = self.uf(tl)
                bt_i01 = self.uf(tr)
                bt_i05 = 0.5 * (bt_i00 + bt_i01)
                # heater temperatures
                ht_i00 = self.kf(tl)
                ht_i01 = self.kf(tr)
                ht_i05 = 0.5 * (ht_i00 + ht_i01)
                dt = tr - tl
                ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
            k_f = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
            k_s = 0.01*k_f
            k_f *= 0.8
            init_tf = self.uf(self.times[0])
            init_ts = init_tf
            return (alpha, init_tf, init_ts, k_f, k_s, toffset)
        else:
            raise ValueError("TwoSideIsolated: not set to prepare to emit")

    def prepare_to_emit(self, *params) -> tuple[ndarray, ndarray]:
        if self.times is not None and self.integrator is not None:
            ts: list[float] = []
            vals: list[ndarray] = []
            # notify(f"preparing to emit for times {self.times}")
            for t in self.times:
                ts.append(t)
                vals.append(self.integrator(t, *params))
            return (array(ts), array(vals))
        else:
            raise ValueError("StiffOfset: not set to prepare to emit")

    def emit_all(self):
        ts, vals = self.prepare_to_emit(*self.params)
        for i in range(len(ts)):
            v = vals[i]
            try:
                v[0]
            except IndexError:
                v = [v]
            # notify(f"printing {ts[i]} and {vals[i]}")
            print(ts[i], *tuple(v))

    def dictonize(self):
        res = dict()
        for k, v in zip(['alpha', 't0_f', 't0_s', 'k_f', 'k_s', 'toffset'], self.params):
            res[k] = v
        return res


class StiffSquareOffset(StiffOffset):
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("stiff_square_offset", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    class IntegrAdapter(Integrator):
        def set_params(self, alpha, init_tf, init_ts, k_f, k_s, toffset):
            ic = array([init_tf, init_ts])
            params = (alpha, k_f, k_s, toffset)
            if (self.ic != ic).any() or self.params != params:
                self.ic = ic
                self.params = params
                self.integrate()

        def __call__(self, arg, *params):
            self.set_params(*params)
            alpha = params[0]
            try:
                return np.sqrt(alpha * (self.pf(arg)[:, 0])**2 +\
                               (1-alpha) * (self.pf(arg)[:, 1])**2
                            )
            except IndexError:
                return np.sqrt(alpha * (self.pf(arg)[0])**2 + \
                               (1-alpha) * (self.pf(arg)[1])**2
                            )
                pass


class StiffBasinOffset(StiffOffset):
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("stiff_basin_offset", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            """ fast_temp = ts[0]
                slow_temp = ts[1]
                """
            # heater_temp = self.kf(time)
            return array((k_f*(heater_temp + toffset - ts[0]),
                          k_s*(heater_temp + toffset - ts[1]),
                          ))
        def _jac(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            return array([[-k_f, 0  ],
                          [ 0,  -k_s],
                          ])
        ############## alpha,   fast_temp,     slow_temp,    k_f,  k_s,  offset
        bounds = (array([0, self.uf(t0)*0.8, self.uf(t0)*0.8, 0,    0,   -ninf]), 
                  array([1, self.uf(t0)*1.2, self.uf(t0)*1.2, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        init_tf = self.uf(t0)
        init_ts = self.uf(t0)
        self.integrator = StiffOffset.IntegrAdapter(
                                                    rhs, jac, self.times,
                                                    array([init_tf, init_ts,]),
                                                    )
        guess = self.generate_guess()
        def of(params):
            return np.var(self.uf(self.times) - self.integrator(self.times, *tuple(params)), ddof=0)
        ores = basinhopping(of, x0=guess)
        # self.params, pcov = curve_fit(self.integrator,
        #                            unknown_fn.table.keys, unknown_fn.table.vals,
        #                            guess,
        #                            bounds=bounds,
        #                            # method='dogbox',
        #                            # jac='3-step'
        #                            )
        # self.params = tuple(self.params)
        self.params = tuple(ores.x)
        # notify(f"sigma: {np.sqrt(np.diag(pcov))}", file=stderr)
        notify("BasinHoppingOffset:", "success" if ores.success else "fail")
        return self

class StiffPenaltyOffset(StiffOffset):
    penalty = 1e+10
    def __init__(self, known_fn: PseudoFunction = None,
                       unknown_fn: PseudoFunction = None,
                       successor = None):
        self.null_init("stiff_penalty_offset", known_fn, unknown_fn, successor)
        if (self.kf is not None and self.uf is not None):
            self.set(known_fn, unknown_fn)

    def set(self, known_fn, unknown_fn):
        self.kf = known_fn
        self.uf = unknown_fn
        self.times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = self.times[0]
        def _rhs(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            """ fast_temp = ts[0]
                slow_temp = ts[1]
                """
            # heater_temp = self.kf(time)
            return array((k_f*(heater_temp + toffset - ts[0]),
                          k_s*(heater_temp + toffset - ts[1]),
                          ))
        def _jac(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            return array([[-k_f, 0  ],
                          [ 0,  -k_s],
                          ])
        ############## alpha,   fast_temp,     slow_temp,    k_f,  k_s,  offset
        bounds = (array([0, self.uf(t0)*0.8, self.uf(t0)*0.8, 0,    0,   -ninf]), 
                  array([1, self.uf(t0)*1.2, self.uf(t0)*1.2, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, self.kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, self.kf(t), *params)
        # notify(f"integrator:\nknown\n{known_fn.table}\nunknown\n{unknown_fn.table}\ntimes\n{self.times}")
        init_tf = self.uf(t0)
        init_ts = self.uf(t0)
        self.integrator = StiffOffset.IntegrAdapter(
                                                    rhs, jac, self.times,
                                                    array([init_tf, init_ts,]),
                                                    )
        guess = self.generate_guess()
        # step 1: curve fit
        self.params, pcov = curve_fit(self.integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   # method='dogbox',
                                   # jac='3-step'
                                   )
        self.params = tuple(self.params)
        # step 2: penalty for
        def of(params):
            mse = np.mean(abs(self.uf(self.times) - self.integrator(self.times, *tuple(params))))
            return mse + StiffPenaltyOffset.penalty*params[-2]**2 # params[-2] == k_s
        _bounds = tuple((x, y) for x, y in zip(bounds[0], bounds[1]))
        notify(f"penalty before minimization of k_s: {of(self.params)}")
        ores = so.minimize(of, x0=self.params, bounds=_bounds, jac='2-point')
        self.params = tuple(ores.x)
        notify(f"penalty after minimization of k_s: {of(self.params)}")
        # notify(f"sigma: {np.sqrt(np.diag(pcov))}", file=stderr)
        notify("StiffPenaltyOffset:", "success" if ores.success else "fail", file=stderr)
        notify(f"StiffPenaltyOffset::message: {ores.message}")
        return self


class Predictors:
    @classmethod
    def get_predictor(cls, kf, uf, mode):
        tp = TempPredictor()
        top = TempOffsetPredictor(successor=tp)
        _2isol = TwoSideIsolated(successor=top)
        _2isol_offset = TwoSideIsolatedOffset(successor=_2isol)
        stiff_offset = StiffOffset(successor=_2isol_offset)
        stiff_square_offset = StiffSquareOffset(successor=stiff_offset)
        stiff_basin_offset = StiffBasinOffset(successor=stiff_square_offset)
        stiff_penalty_offset = StiffPenaltyOffset(successor=stiff_basin_offset)
        chain = stiff_penalty_offset
        # chain = tp.connect(top.connect)
        predictor = chain.find(mode)
        # if predictor is not None:
        #     notify(f"found {mode} predictor to be {predictor.mode}", file=stderr)
        # return chain.find(mode).set(kf, uf)
        return predictor.set(kf, uf)


if __name__ == "__main__":
    main()

