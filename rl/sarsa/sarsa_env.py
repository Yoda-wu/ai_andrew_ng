import numpy as np
from collections import defaultdict

class WindyGridWorld():

    def __init__(self, isEight = False, isNinth = False) -> None:
        self.max_x = 10
        self.max_y = 7
        self.target_x = 8
        self.target_y = 4
        self.action_spaces = 4
        if isEight:
            self.action_spaces = 8
            self.isEight = isEight
        if isNinth:
            self.action_spaces = 9
            self.isNinth = isNinth
        self.wind_dict =  {
            1:0, 2:0,3:0,
            4:1, 5:1, 6:1,
            7:2, 8:2, 9:1,
            10:0
        }
    
    def step(self, pos, action):
        x, y = pos
        wind = self.wind_dict[x]

        if action == 0 : # left
            next_state = max(1, x-1), min(self.max_y, y+wind)
        elif action == 1: # right
            next_state = min(self.max_x, x + 1) , min(self.max_y, y+wind)
        elif action == 2 : # up
            next_state = x, min(self.max_y, y+wind+1)
        elif action == 3 : # down
            next_state = x, max(0, min(self.max_y, y+wind-1))
        else:
            if self.isEight:
                if action == 4 : # left - up
                    next_state = max(1, x-1), min(self.max_y, y+wind+1)
                elif action == 5 : # left - down
                    next_state = max(1, x-1), max(0, min(self.max_y, y+wind-1))
                elif action == 6 : # right - up
                    next_state = min(self.max_x, x + 1) , min(self.max_y, y+wind+1)
                elif action == 7 : # right - down
                    next_state = min(self.max_x, x + 1) , max(0, min(self.max_y, y+wind-1))
                else:
                    if self.isNinth:
                        if action == 8 : # stay
                            next_state = x, min(self.max_y, y+wind)
                        else:
                            raise ValueError
                    else:
                        raise ValueError
            else:
                raise ValueError
        if next_state == (self.target_x, self.target_y):
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        return next_state, reward, done, None


        
        
    