from collections import defaultdict

tan = [
    ('s1','a1','r1','ss1',False),
     ('s2','a2','r2','ss2',False),
      ('s3','a3','r3','ss3', True),
       ]

s, a, r, ns, d = zip(*tan)

print(s)
print(a)
print(r)
print(ns)
print(d)


import numpy as np

print(np.array(s))