from scipy.optimize import linear_sum_assignment

import numpy as np
import util 
                #  0, 1, 2,3, 4, 5
cost = np.array([ [4, 1, 3,4, 1, 3 ],# 0
                  [2, 0, 5,2, 0, 5], # 1
                  [3, 2, 2,3, 2, 2], # 2
                  [1, 7, 4,1, 7, 4], # 3
                  [6, 3, 0,6, 3, 0], # 4
                  [2, 4, 5,2, 4, 5]  # 5
                 ])

row_ind, col_ind = linear_sum_assignment(cost)
for i in range(10):
    print(util.get_cnt())


