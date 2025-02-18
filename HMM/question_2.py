import grid as gd 
import matplotlib.pyplot as plt
import numpy as np
import helperFunctions as hf


def get_TB():
  T_true = np.zeros((225, 225))
  for i in range(225):
    for j in range(225):
      T_true[i][j]= hf.get_transition_prob(hf.get_state(i), hf.get_state(j))

  B_true = np.zeros((225, 16))
  for i in range(225):
    for j in range(16):
      B_true[i][j]=hf.get_observation_prob(hf.get_state(i), hf.get_observation(j))
  return T_true, B_true

def forward_inference(obs):
  T, B = get_TB()
  #alpha
  alpha = np.zeros((225, 30))

  #filling first column
  for s in range (0,15*15):
    if s==0:
      alpha[s][0]=1*B[s][hf.get_observation_number(obs[0])]

  #filling remaining columns
  for o_number in range (1, len(obs)):
    o=hf.get_observation_number(obs[o_number])
    for s in range(0, 15*15):
      for s_prev in range(0,15*15):
        alpha[s][o_number] += (alpha[s_prev][o_number-1]*T[s_prev][s]*B[s][o])

  return alpha


#E0.00000000000031781888
#0.00000000000000018443