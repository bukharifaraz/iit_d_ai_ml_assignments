import grid as gd 
import matplotlib.pyplot as plt
import numpy as np
import helperFunctions as hf
import random

   
# Generating Grid Images for four sensors
def generate_grid_image_for_sensors():
    sensor_obs_prob=hf.sensor_observation_probability()
    img = []
    for sensor in range(1,5):
        grid = gd.Grid(sensor_obs_prob[sensor])
        grid.show("sensor_obs_prob.png")
        img.append(plt.imread("sensor_obs_prob.png"))

    # displaying the observation probabilities
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    axs[1][0].imshow(img[0])
    axs[1][0].set_title ("Sensor 1")

    axs[0][0].imshow(img[1])
    axs[0][0].set_title ("Sensor 2")

    axs[0][1].imshow(img[2])
    axs[0][1].set_title ("Sensor 3")

    axs[1][1].imshow(img[3])
    axs[1][1].set_title ("Sensor 4")


    for i in range(2):
        for j in range(2):
            axs[i][j].set_xticks([0, img[1].shape[1]//3, 2*img[1].shape[1]//3, img[1].shape[1]-1])
            axs[i][j].set_xticklabels([1, 5, 10, 15])
            axs[i][j].set_yticks([0, img[1].shape[0]//3, 2*img[1].shape[0]//3, img[1].shape[0]-1])
            axs[i][j].set_yticklabels([15, 10, 5, 1])

    plt.show()
    
    
    
#-----------------------   part b ---------------------

def get_up_down_left_right_prob(current_state):
  #0 => right
  #1 => up
  #2 => down
  #3 => left
  #4 => same
  p=[0.4, 0.3, 0.1, 0.1, 0.1]
  if current_state[1]==1: #bottom-most row
    p[4]=p[4]+p[2]
    p[2]=0
  if current_state[1]==15: #top-most row
    p[4]=p[4]+p[1]
    p[1]=0
  if current_state[0]==1: #left-most column
    p[4]=p[4]+p[3]
    p[3]=0
  if current_state[0]==15: #right-most column
    p[4]=p[4]+p[0]
    p[0]=0
  return p


def sample_one_state_sequence(sample_length, seed_value):
  current_state = [1,1]
  traj = []
  traj.append(current_state[:])
  np.random.seed(seed_value)
  for i in range(sample_length-1):
    state_change = np.random.choice(np.arange(0, 5), p=get_up_down_left_right_prob(current_state))
    if state_change == 0:
      current_state[0]=current_state[0]+1
    if state_change == 1:
      current_state[1]=current_state[1]+1
    if state_change == 2:
      current_state[1]=current_state[1]-1
    if state_change == 3:
      current_state[0]=current_state[0]-1
    traj.append(current_state[:])
  return traj

def sample(n_samples, sample_length, seed_value=10):
  np.random.seed(seed_value)
  sensor_obs_prob = hf.sensor_observation_probability()
  #sampling state_trajectories
  state_sequences = []
  for i in range(n_samples):
    seed_value = seed_value+1
    t = sample_one_state_sequence(sample_length, seed_value)
    state_sequences.append(t[:])

  #sampling #n_samples observations of length #sample_length each
  observation_sequences = []
  for t in range(n_samples):
    observation_sequence = []
    for s in state_sequences[t]:
      #print('state is', s)
      #finding observation probabilities of i-th sensors at this state:
      sp=[]
      for i in range(1,5):
        sp.append(round(sensor_obs_prob[i][15-s[1]][s[0]-1],2))
      #finding observation of 4 sensors at this state
      observation = []
      for i in range (0,4):
        o = np.random.choice(np.arange(0, 2), p=[1-sp[i], sp[i]])
        observation.append(int(o))
      #print('State is ', s, 'Sensor probs are', sp, '    Sampled Observation is', observation)
      observation_sequence.append(observation[:])
    observation_sequences.append(observation_sequence[::])
  return state_sequences, observation_sequences
    
    
    
#------------------------ c -------------------------------

def plot_sampled_trajectories(state_sequences):
    sensor_obs_prob = hf.sensor_observation_probability()
    img = []
    for t in state_sequences:
        # Create an instance of the Grid class
        grid = gd.Grid(sensor_obs_prob[0]) #parameter to give white color to background
        grid.show("output_grid.png")
        # Define a path to draw (sequence of (x, y) tuples)
        path = t
        # Draw the path on the grid
        grid.draw_path(path, color=[1, 0, 0])  # Red color for the path
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

