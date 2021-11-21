from hp import *

def f(t, x, w1, w2):
    return array([w1*x[1], -w2*x[0]])

def jac(f, x, w1, w2):
    return array([[0,   w1],
                  [-w2, 0]])

times = array(range(20))
integrator = Integrator(f, jac, array([1, 0]), times)


for t in times:
    print(integrator(t, 0.4, 0.4))
