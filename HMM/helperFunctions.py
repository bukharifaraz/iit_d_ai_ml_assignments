import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#mapping indices of Python Matrix with Physical Grid
def index_mapping():
    x=[0]
    y=[0]
    for i in range(1, 16):
        x.append(i-1)
    for i in range(1, 16):
        row = 15 - i
        y.append(row)  
    return x, y

#states are from 0 to 224 corresponding to (1,1), (1,2),..,(15,4),(15,15)
def get_state_number(state):
  return (((state[0]-1)*15)+state[1]-1)


#returns [x,y] corresponding to state number
def get_state(state_number):
  x = (state_number//15)
  y=(state_number-x*15)
  #returns (x,y) corresponding to a state_number
  return [x+1,y+1]

#observations are numbered from 0 to 15 (4bit binary to decimal conversion)
def get_observation_number(obs):
  return (obs[3]+(obs[2]*2)+(obs[1]*4)+(obs[0]*8))

#returns 4-bit observation corresponding to the observation number
def get_observation(obs_number):
  observation = []
  for i in range(4):
    d = obs_number % 2
    obs_number = int(obs_number // 2)
    observation.append(d)
  return list(reversed(observation))

# filling observation probability matrices for individual sensors
# sensor_obs_prob[k][x][y] gives the probability of observation of sensor k, at location (x, y) on the Grid
def sensor_observation_probability():
    x, y = index_mapping()
    
    sensor_obs_prob = np.zeros((5, 15, 15))
    sensor_obs_prob[1]
    for j in range (1,10):
        for i in range(1, 10):
            sensor_obs_prob[1][y[j], x[i]] = (18 - (i-1) - (j-1))/18

    for j in range (7,16):
        for i in range(1, 10):
            sensor_obs_prob[2][y[j], x[i]] = (18 - (i-1) + (j-15))/18

    for j in range (7,16):
        for i in range(7, 16):
            sensor_obs_prob[3][y[j], x[i]] = (18 + (i-15) + (j-15))/18

    for j in range (1, 10):
        for i in range(7, 16):
            sensor_obs_prob[4][y[j], x[i]] = (18 + (i-15) - (j-1))/18
    return sensor_obs_prob


#receives previous and current states
#returns the transition probability from previous to current state
def get_transition_prob(previous_state, current_state):
  #0 => right
  #1 => up
  #2 => down
  #3 => left
  #4 => same
  p=[0.4, 0.3, 0.1, 0.1, 0.1]
  if previous_state[1]==1: #bottom-most row
    p[4]=p[4]+p[2]
    p[2]=0
  if previous_state[1]==15: #top-most row
    p[4]=p[4]+p[1]
    p[1]=0
  if previous_state[0]==1: #left-most column
    p[4]=p[4]+p[3]
    p[3]=0
  if previous_state[0]==15: #right-most column
    p[4]=p[4]+p[0]
    p[0]=0

  horizontal_shift = current_state[0]-previous_state[0]
  vertical_shift   = current_state[1]-previous_state[1]

  if(horizontal_shift==1 and vertical_shift==0): #i.e. right
    return p[0]
  elif(horizontal_shift==-1 and vertical_shift==0):#i.e. left
    return p[3]
  elif(vertical_shift==1 and horizontal_shift==0):#i.e. up
    return p[1]
  elif(vertical_shift==-1 and horizontal_shift==0):#i.e. down
    return p[2]
  elif(horizontal_shift==0 and vertical_shift==0):#i.e. same
    return p[4]
  else:
    return 0


#receives observation and current states
#returns the probability of observation in the given state
def get_observation_prob(state, observation):
  s=1
  p=1
  sensor_obs_prob=sensor_observation_probability()
  for o in observation:
    #print(round(sensor_obs_prob[s][15-state[1]][state[0]-1],2))
    p=p*(o*sensor_obs_prob[s][15-state[1]][state[0]-1]) + p*(1-o)*(1-sensor_obs_prob[s][15-state[1]][state[0]-1])
    s=s+1
  return p

#returns dictionary containing the list of neighbours to all states
def get_neighbours():
  neighbours={}
  for i in range(225):
    neighbours[i]=[i]
    s = get_state(i)
    x=s[0]
    y=s[1]
    if(x-1>0):
      neighbours[i].append(get_state_number([x-1,y]))
    if(x+1<16):
      neighbours[i].append(get_state_number([x+1,y]))
    if(y-1>0):
      neighbours[i].append(get_state_number([x,y-1]))
    if(y+1<16):
      neighbours[i].append(get_state_number([x,y+1]))
  return neighbours
      
    