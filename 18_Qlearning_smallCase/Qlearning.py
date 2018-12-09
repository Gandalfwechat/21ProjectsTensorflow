import random
import numpy as np 
from env import Env
import time 

e = Env()
Q = np.zeros((e.state_num,4))
# then started to train the agent (update the Q matrix) and play the game 
MAX_STEP = 40
ALPHA = 0.2
GAMMA = 0.75
Epsilon = 0.1


def epsilon_greedy(Q,indx):
    prob = random.random()
    action = 99; 
    if prob>0.2: 
        action = np.argmax(Q[indx,:])
    else:
        action = np.random.randint(0,4) # generate integers : 0, 1, 2, 3
    return action 


for i in range(1):
    e = Env() 
    while (e.is_end is False)  and (e.step<MAX_STEP):      
        action = epsilon_greedy(Q,e.present_state)
        state = e.present_state
        #print('action: ',action,'state: ', state,'step:',e.step )
        reward = e.interact(action)
        new_state = e.present_state
        Q[state,action] = (1-ALPHA)*Q[state,action]+ ALPHA*(reward + GAMMA*Q[new_state,:].max())
        e.print_map()
    print('eps: ',i,'total step: ',e.step,' total reward ',e.total_reward)   
    






















