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

def viterbi(obs):
  #v
  v  = []
  bt = []
  T , B = get_TB()
  #filling first column of v and bt 
  for s in range (0,15*15):
    if s==0:
      v.append([1*hf.get_observation_prob(hf.get_state(s),obs[0])])
    else:
      v.append([0])
    bt.append([0])
  #print('v is :', v)
  #print('bt is : ', bt)

  #filling remaining columns
  for o_number in range (1, len(obs)):
    o = hf.get_observation_number(obs[o_number])
    #print('Filling column', o_number, ' of v and bt ')
    for s in range(0, 15*15): # fill v and bt for this state
      #print('Calculations started for state: ', s)
      max_pp_tp_ep = 0
      bt_state = -1

      for s_prev in range(0,15*15):
        tp = T[s_prev][s] #hf.get_transition_prob(hf.get_state(s_prev),hf.get_state(s))
        ep = B[s][o] #hf.get_observation_prob(hf.hf.get_state(s), obs[o_number])
        pp = v[s_prev][o_number-1]
        if(pp * tp * ep > max_pp_tp_ep):
          #if(o_number==29 and s==186):
            #print('preceding state to last state is : ',get_state(s_prev))
          max_pp_tp_ep=pp * tp * ep
          bt_state=s_prev
      v[s].append(max_pp_tp_ep)
      #if(o_number==29 and s==186):
          #print('preceding state to last state is : ',get_state(v[s][28]))
      bt[s].append(bt_state)
  #return v, bt

  # getting the start of backtrack (i.e. the last state)
  s=[]
  max_p=0
  last_state=-1
  for i in range(15*15):
    if v[i][29]>max_p:
      max_p=v[i][29]
      last_state=i #last state
  s.append(hf.get_state(last_state))
  #print('the last state obtained is: ', get_state(last_state))
  #print('the prob of last state  is: ', v[last_state][29])

  for i in range(1,len(obs)):
    o_num = len(obs) - i
    #print('last_state is ', last_state, ' o_num is ', o_num, end = ' ')
    #print()
    last_state = bt[last_state][o_num]
    s.append(hf.get_state(last_state))
  s=list(reversed(s))
  return s #,v, bt;




def plot_decoded_trajectories(state_sequences, most_likely_state_sequence):
  img = []
  for t in range(len(state_sequences)):
    # Create an instance of the Grid class
    grid = gd.Grid(hf.sensor_observation_probability()[0]) #to give white background color
    grid.show("output_grid.png")
    # Define a path to draw (sequence of (x, y) tuples)
    path1 = state_sequences[t]
    path2 = most_likely_state_sequence[t]
    # Draw the path on the grid
    grid.draw_path(path1, color=[1, 0, 0])  # Red color for the path
    grid.draw_path(path2, color=[0, 0, 139])  # Red color for the path
    # Show the grid and save it as an image
    grid.show("output_grid.png")
    # Optionally, open the image to display it
    img.append(plt.imread("output_grid.png"))


  # displaying the observation probabilities
  fig, axs = plt.subplots(4, 5, figsize=(15, 15))
  fig.subplots_adjust(wspace=0.3, hspace=0.3)

  k=0
  for i in range(4):
    for j in range(5):
      axs[i][j].set_xticks([0, img[1].shape[1]//3, 2*img[1].shape[1]//3, img[1].shape[1]-1])
      axs[i][j].set_xticklabels([1, 5, 10, 15])
      axs[i][j].set_yticks([0, img[1].shape[0]//3, 2*img[1].shape[0]//3, img[1].shape[0]-1])
      axs[i][j].set_yticklabels([15, 10, 5, 1])

      axs[i][j].imshow(img[k])
      title = 'Trajectory #'+ str(k+1)
      axs[i][j].set_title(title)
      k=k+1

  plt.show()


