from __future__ import print_function 
import copy 

# define 
MAP =\
'''
.........
.       .
.     o .
.       .
.........

'''
MAP = MAP.strip().split('\n')
MAP = [[c for c in line] for line in MAP] # split the MAP into two dimension array 
DX = [-1, 1, 0, 0]
DY = [0 , 0,-1, 1]

class Env(object): 
    def __init__(self):
        self.map = copy.deepcopy(MAP)
        self.x=1
        self.y=1
        self.step = 0
        self.total_reward = 0 
        self.is_end = False 

    def interact(self, action):
        assert self.is_end is False 
        new_x = self.x + DX[action]
        new_y = self.y + DY[action]
        new_position_char = self.map[new_x][new_y]
        self.step+=1
        if new_position_char == '.':
            reward = 0 
        elif new_position_char == ' ':
            self.x = new_x
            self.y = new_y
            reward = 0 
        elif new_position_char == 'o':
            self.x = new_x
            self.y = new_y 
            self.map[new_x][new_y]=' ' # update state and collect the gold !       
            self.is_end = True  # end the game
            reward = 100
        elif new_position_char == 'x':
            self.x = new_x
            self.y = new_y             
            #self.is_end = True
            self.map[new_x][new_y]=' '
            reward = -100
        self.total_reward +=reward
        return reward

    @property 
    def state_num(self):
        rows = len(self.map) 
        cols = len(self.map[0])
        return rows*cols


    @property
    def present_state(self):
        cols = len(self.map[0])
        return self.x*cols+self.y 

    def print_map(self):
        printed_Map = copy.deepcopy(self.map)
        printed_Map[self.x][self.y]='A'
        MapOneStr = '\n'.join( ''.join( c for c in line ) for line in printed_Map ) 
        print(MapOneStr)
