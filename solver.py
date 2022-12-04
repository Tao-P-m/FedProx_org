import numpy as np
from scipy.linalg import lstsq, pinv
from scipy.optimize import linprog

# init standard output
n = 20
m = 5
p = 19
rules = [[2,7,15], [5,19,0], [3,6,1], [6,9,1], [7,19,17]]

stand_out = np.zeros(n, dtype=np.int32)
a_ue = []
b_ue = []
a_e = []
b_e = []
for idx, rule in enumerate(rules):
    l = rule[0]
    r = rule[1]
    k = rule[2]
    rule_arr = np.zeros(n, dtype=np.int32)
    rule_arr[l:r] = 1
    if k > 0:
        a_e.append(rule_arr)
        b_e.append(k+p)
    else:
        a_ue.append(rule_arr)
        b_ue.append(k)
bounds = []
for i in range(n):
    bounds.append((0, None))

#x = lstsq(np.asarray(a), np.array(b), cond=None)
#print(x)
#v = pinv(np.asarray(a)) @ np.array(b)
#print(v)

r = linprog(c=np.ones(n), 
            A_ub=np.asarray(a_ue),
            b_ub=np.asarray(b_ue),
            A_eq=np.asarray(a_e), 
            b_eq=np.array(b_e), 
            bounds=tuple(bounds))
print(r.x)