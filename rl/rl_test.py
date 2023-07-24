from collections import defaultdict

a = defaultdict(lambda: 0)
state =  1,4

a[str(state)] += 1
a[str(state)] += 1
a[str(state)] += 1
print(a[str(state)])
a[str(state)] =22
print(a[str(state)])