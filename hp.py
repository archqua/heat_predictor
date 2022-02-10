#!/bin/env python

import argparse
from sys import stdin, stderr, stdout
from scipy.optimize import curve_fit, basinhopping
import scipy.optimize as so
from scipy.integrate import ode
from scipy.interpolate import interp1d
import numpy as np
from numpy import inf as ninf
from numpy import ndarray, array
from functools import reduce


default_mode = "stiff_penalty_offset"
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
              "stiff_offset",
              "stiff_penalty_offset",
              }
pow_modes: set[str] = set()
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
    predictor = Predictors.get_predictor(mode)
    if predictor is None:
        notify(f"unknown mode {mode}", file=stderr)
        notify("exiting...", file=stderr)
        exit(6)
    predictor.fit(known_fn, unknown_fn)
    params = predictor.dictonize()
    notify(params, file=stderr)
    print_np_array(predictor.predict(known_fn, unknown_fn))

def print_np_array(ar, file=stdout, delim=val_delim):
    # notify(f"prinitng array\n{ar}")
    for i in range(len(ar)):
        print(delim.join(map(str, ar[i])), file=file)

class PseudoFunction:
    def __init__(self, table_dict):
        self.table = Table.FromDict(table_dict)
        # WARNING: out of bounds not handled
        self.lboundary = self.table.keys[0]
        self.rboundary = self.table.keys[-1]
        self.interpolator = interp1d(self.table.keys, self.table.vals,
                                     fill_value="extrapolate", bounds_error=False,
                                     assume_sorted=True,
                                     axis=0,
                                     )

    def __call__(self, arg):
        # WARNING: return type is array, not float
        return self.interpolator(arg)


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
        self.set_params(*params)
        return self.pf(arg)

    def set_params(self, *params):
        if self.params != params[1:] or self.ic != params[0]:
            self.ic = params[0]
            self.params = params[1:]
            self.integrate()

    def integrate(self):
        solver = ode(lambda t, x: self.rhs(t, x, *self.params),
                     lambda t, x: self.jac(t, x, *self.params))
        time2val = dict()
        # WARNING: out of bounds not handled
        t0, tend = self.times[0], self.times[-1]
        time2val[t0] = self.ic
        solver.set_initial_value(self.ic, t0)
        for time in self.times[1::]:
            if solver.successful():
                time2val[time] = solver.integrate(time)
                if (time2val[time].shape == (1,)):
                    time2val[time] = time2val[time].reshape(())
            else:
                break
        self.pf = PseudoFunction(time2val)


class IPredictor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"AbstractPredictor: __init__({args}, {kwargs})")

    def fit(self, *args, **kwargs):
        raise NotImplementedError(f"AbstractPredictor: fit({args, kwargs})")

    def predict(self, *args, **kwargs):
        raise NotImplementedError(f"AbstractPredictor: predict({args, kwargs})")

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


class TempPredictor(IPredictor):
    def __init__(self, successor = None):
        self.mode = "temp"
        self.successor = successor
        self.integrator = None
        self.data = dict()

    def fit(self, known_fn, unknown_fn):
        integrator = self.make_integrator(known_fn, unknown_fn)
        guess, bounds = self.generate_guess_bounds(known_fn, unknown_fn)
        params, _ = curve_fit(integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        params = tuple(params)
        self.data['params'] = params
        return self

    @staticmethod
    def make_integrator(known_fn, unknown_fn):
        kf = known_fn
        uf = unknown_fn
        times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = times[0]
        def _rhs(time, body_temp, heater_temp, kappa, kappa0):
            # heater_temp = kf(time)
            return kappa*(heater_temp - body_temp) + \
                   kappa0*(outside_temp - body_temp)
        def _jac(time, body_temp, heater_temp, kappa, kappa0):
            return -kappa - kappa0
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, kf(t), *params)
        return Integrator(rhs, jac, times, unknown_fn(t0))

    @staticmethod
    def generate_guess_bounds(known_fn, unknown_fn) -> tuple:
        k0 = 0
        ks: list[float] = []
        times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = times[0]
        for tl, tr in zip(times[:min(9, len(times) - 1)],
                          times[1:min(10, len(times))]):
            # body temperatures
            bt_i00 = unknown_fn(tl)
            bt_i01 = unknown_fn(tr)
            bt_i05 = 0.5 * (bt_i00 + bt_i01)
            # heater temperatures
            ht_i00 = known_fn(tl)
            ht_i01 = known_fn(tr)
            ht_i05 = 0.5 * (ht_i00 + ht_i01)
            dt = tr - tl
            ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
        k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
        guess = (unknown_fn(t0), k, k0)
        bounds = (array([unknown_fn(t0)*0.9, 0, 0]), 
                  array([unknown_fn(t0)*1.1, ninf, ninf]),
                  )
        return (guess, bounds)

    def predict(self, known_fn, unknown_fn):
        times = known_fn.table.submerged_keys(unknown_fn.table)
        integrator = self.make_integrator(known_fn, unknown_fn)
        # WARNING unchecked 'params' key
        vals = integrator(times, *self.data['params'])
        # notify(f"stacked to shape {np.hstack((np.array(times)[:,np.newaxis], vals[:,np.newaxis])).shape}")
        return np.hstack((np.array(times)[:,np.newaxis], vals[:,np.newaxis]))

    def dictonize(self):
        res = dict()
        params = self.data['params']
        res['t0'] = params[0]
        res['k']  = params[1]
        res['k0'] = params[2]
        return res


class TempOffsetPredictor(TempPredictor):
    def __init__(self, successor = None):
        self.mode = "temp_offset"
        self.successor = successor
        self.integrator = None
        # THIS ANNOTATION LIES
        self.data: dict[str, int] = dict()

    def fit(self, known_fn, unknown_fn):
        integrator = self.make_integrator(known_fn, unknown_fn)
        guess, bounds = self.generate_guess_bounds(known_fn, unknown_fn)
        params, _ = curve_fit(integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        params = tuple(params)
        self.data['params'] = params
        return self

    @staticmethod
    def generate_guess_bounds(known_fn, unknown_fn) -> tuple:
        k0 = 0
        ks: list[float] = []
        toffset = 0
        times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = times[0]
        for tl, tr in zip(times[:min(9, len(times) - 1)],
                          times[1:min(10, len(times))]):
            # body temperatures
            bt_i00 = unknown_fn(tl)
            bt_i01 = unknown_fn(tr)
            bt_i05 = 0.5 * (bt_i00 + bt_i01)
            # heater temperatures
            ht_i00 = known_fn(tl)
            ht_i01 = known_fn(tr)
            ht_i05 = 0.5 * (ht_i00 + ht_i01)
            dt = tr - tl
            ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
        k = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
        guess = (unknown_fn(t0), k, k0, toffset)
        bounds = (array([unknown_fn(t0)*0.9, 0,    0,   -ninf]), 
                  array([unknown_fn(t0)*1.1, ninf, ninf, ninf]),
                  )
        return (guess, bounds)

    @staticmethod
    def make_integrator(known_fn, unknown_fn):
        kf = known_fn
        uf = unknown_fn
        times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = times[0]
        def _rhs(time, body_temp, heater_temp, kappa, kappa0, toffset):
            # heater_temp = kf(time)
            return kappa*(heater_temp + toffset - body_temp) + \
                   kappa0*(outside_temp - body_temp)
        def _jac(time, body_temp, heater_temp, kappa, kappa0, toffset):
            return -kappa - kappa0
        def rhs(t, x, *params):
            # notify("rhs call for at", t, "for", x, "with", *params)
            return _rhs(t, x, kf(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, kf(t), *params)
        return Integrator(rhs, jac, times, unknown_fn(t0))

    # predict is inherited

    def dictonize(self):
        res = dict()
        res['t0']      = self.data['params'][0]
        res['k']       = self.data['params'][1]
        res['k0']      = self.data['params'][2]
        res['toffset'] = self.data['params'][3]
        return res


class StiffOffset(IPredictor):
    def __init__(self, successor = None):
        self.mode = "stiff_offset"
        self.successor = successor
        self.integrator = None
        self.data = dict()

    def fit(self, known_fn, unknown_fn):
        integrator = self.make_integrator(known_fn, unknown_fn)
        guess, bounds = self.generate_guess_bounds(known_fn, unknown_fn)
        params, pcov = curve_fit(integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        params = tuple(params)
        self.data['params'] = params
        notify(f"sigma: {np.sqrt(np.diag(pcov))}", file=stderr)
        return self

    @staticmethod
    def make_integrator(known_fn, unknown_fn):
        times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = times[0]
        def _rhs(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            """ fast_temp = ts[0]
                slow_temp = ts[1]
                """
            return array((k_f*(heater_temp + toffset - ts[0]),
                          k_s*(heater_temp + toffset - ts[1]),
                          ))
        def _jac(time, ts, heater_temp, alpha, k_f, k_s, toffset):
            return array([[-k_f, 0  ],
                          [ 0,  -k_s],
                          ])
        ############## alpha,   fast_temp,         slow_temp,      k_f,  k_s,  offset
        bounds = (array([0, unknown_fn(t0)*0.8, unknown_fn(t0)*0.8, 0,    0,   -ninf]), 
                  array([1, unknown_fn(t0)*1.2, unknown_fn(t0)*1.2, ninf, ninf, ninf]),
                  )
        def rhs(t, x, *params):
            return _rhs(t, x, known_fn(t), *params)
        def jac(t, x, *params):
            return _jac(t, x, known_fn(t), *params)
        init_tf = unknown_fn(t0)
        init_ts = unknown_fn(t0)
        return StiffOffset.IntegrAdapter(
                                         rhs, jac, times,
                                         array([init_tf, init_ts,]),
                                         )
        guess = self.generate_guess()
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

    @staticmethod
    def generate_guess_bounds(known_fn, unknown_fn) -> tuple:
        times = known_fn.table.submerged_keys(unknown_fn.table)
        t0 = times[0]
        alpha = 0.85 # good one
        # alpha = 0.8 # worse one
        toffset = 0
        ks: list[float] = []
        for tl, tr in zip(times[:min(9, len(times) - 1)],
                          times[1:min(10, len(times))]):
            # body temperatures
            bt_i00 = unknown_fn(tl)
            bt_i01 = unknown_fn(tr)
            bt_i05 = 0.5 * (bt_i00 + bt_i01)
            # heater temperatures
            ht_i00 = known_fn(tl)
            ht_i01 = known_fn(tr)
            ht_i05 = 0.5 * (ht_i00 + ht_i01)
            dt = tr - tl
            ks.append((bt_i01 - bt_i00) / (dt * (ht_i05 - bt_i05)))
        k_f = reduce(lambda l, r: l + r, ks, 0.0) / len(ks)
        k_s = 0.01*k_f
        k_f *= 0.8
        init_tf = unknown_fn(times[0])
        init_ts = init_tf
        guess = (alpha, init_tf, init_ts, k_f, k_s, toffset)
        ############## alpha,   fast_temp,         slow_temp,      k_f, k_s, offset
        bounds = (array([0, unknown_fn(t0)*0.8, unknown_fn(t0)*0.8, 0,    0, -3]), 
                  array([1, unknown_fn(t0)*1.2, unknown_fn(t0)*1.2, 1, 1e-03, 3]),
                  )
        return (guess, bounds)

    def predict(self, known_fn, unknown_fn):
        times = known_fn.table.submerged_keys(unknown_fn.table)
        integrator = self.make_integrator(known_fn, unknown_fn)
        # WARNING unchecked 'params' key
        vals = integrator(times, *self.data['params'])
        # notify(f"stacked to shape {np.hstack((np.array(times)[:,np.newaxis], vals[:,np.newaxis])).shape}")
        return np.hstack((np.array(times)[:,np.newaxis], vals[:,np.newaxis]))

    def dictonize(self):
        res = dict()
        for k, v in zip(['alpha', 't0_f', 't0_s', 'k_f', 'k_s', 'toffset'], self.data['params']):
            res[k] = v
        return res


class StiffPenaltyOffset(StiffOffset):
    def __init__(
            self,
            # version minimize
            penalty=4e+09,
            # version least_squares
            # penalty=4e+03,
            successor = None,
            ):
        self.mode = "stiff_penalty_offset"
        self.successor = successor
        self.integrator = None
        self.data = dict()
        self.penalty = penalty

    def fit(self, known_fn, unknown_fn):
        integrator = self.make_integrator(known_fn, unknown_fn)
        guess, bounds = self.generate_guess_bounds(known_fn, unknown_fn)
        params, pcov = curve_fit(integrator,
                                   unknown_fn.table.keys, unknown_fn.table.vals,
                                   guess,
                                   bounds=bounds,
                                   )
        params = tuple(params)
        # version minimize
        def of(params):
            times = unknown_fn.table.keys
            prediction = integrator(times, *params)
            return np.mean((prediction - unknown_fn(times))**2) +\
                   self.penalty*params[-2]**2 + self.penalty*(params[0]/1000000)**2
        # version least_squares
        # def of(params):
        #     times = unknown_fn.table.keys
        #     prediction = integrator(times, *params)
        #     return np.hstack((prediction - unknown_fn(times), len(times)*self.penalty*params[-2]))
        def qf(params):
            times = known_fn.table.keys
            times = times[-len(times)//4:]
            prediction = integrator(times, *params)
            return np.mean((prediction - known_fn(times))**2)
        notify("before penalty")
        notify(f"sigma: {np.sqrt(np.diag(pcov))}", file=stderr)
        ofval = of(params)
        notify(f"objective function value: {of(params)}")
        notify(f"quality function value: {qf(params)}")
        _bounds = tuple((x, y) for x, y in zip(bounds[0], bounds[1]))
        # version minimize
        ores = so.minimize(of, x0=params, bounds=_bounds, jac='2-point')
        # version least_squares
        # ores = so.least_squares(of, x0=params, bounds=bounds)
        params = tuple(ores.x)
        notify("after penalty", self.penalty)
        notify(f"objective function value: {of(params)}")
        notify(f"quality function value: {qf(params)}")
        notify("StiffPenaltyOffset:", "success" if ores.success else "fail")
        notify(f"StiffPenaltyOffset::message: {ores.message}")
        self.data['params'] = params
        return self



class Predictors:
    @classmethod
    def get_predictor(cls, mode):
        tp = TempPredictor()
        top = TempOffsetPredictor(successor=tp)
        stiff_offset = StiffOffset(successor=top)
        stiff_penalty_offset = StiffPenaltyOffset(successor=stiff_offset)
        chain = stiff_penalty_offset
        predictor = chain.find(mode)
        return predictor

if __name__ == "__main__":
    main()
