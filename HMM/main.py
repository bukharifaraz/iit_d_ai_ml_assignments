import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import grid as gd
import helperFunctions as hf
import question_1 as q1
import question_2 as q2
import question_3 as q3
import question_4 as q4


# 1.  Sample Trajectories & Observations #************************

# a 
q1.generate_grid_image_for_sensors()


# b 
state_sequences, observation_sequences = q1.sample(20, 30)
i=1
for state_sequence in state_sequences:
  print('\nState Sequence#',i)
  for s in state_sequence:
      print((s[0],s[1]), end= ' ')
  i=i+1


# c 
q1.plot_sampled_trajectories(state_sequences)


# d 
for i in range(20):
    print('\nObservation sequence #', i+1)
    print(observation_sequences[i])


#*********************  2.  Likelihood Estimation (Forward Algorithm) #*********************


for observation in observation_sequences:
  alpha = q2.forward_inference(observation)
  likelihood=0
  for i in range(225):
    likelihood+=alpha[i][29]
  likelihood=likelihood
  print('\nLikelihood of this observation sequence is:\n', observation, f"{likelihood:.4e}")
  print(likelihood)
  

  


#***************************  3. Decoding (Viterbi Algorithm) #***************************


# a 

i=0
#state sequences
most_likely_state_sequence = []
print('\n\nCreation of Viterbi tables in progress ....')
for observation_sequence in observation_sequences:
  most_likely_state_sequence.append(q3.viterbi(observation_sequence))


print('\n\nViterbi Results\n\n')
for i in range(len(most_likely_state_sequence)):
  print('Sampled state sequence #', i+1, ' is: ')
  for s in state_sequences[i]:
    print(s, ' ', end=' ')
  print('\nMost probable state sequence is :')
  for s in most_likely_state_sequence[i]:
    print(s, ' ', end=' ')
  i+=1
  


# b 

avg_dis = []
for t in range(30):
  dis = 0
  for i in range(20):
    known_state_sequence = state_sequences[i][t]
    predicted_state_sequence = most_likely_state_sequence[i][t]
    dis+=abs(known_state_sequence[0]-predicted_state_sequence[0])+abs(known_state_sequence[1]-predicted_state_sequence[1])
  avg_dis.append(dis/20)
 
print('\n\nAverage Distances vs Time are:') 
print(avg_dis)
plt.plot(avg_dis)
  
  

# c
q3.plot_decoded_trajectories(state_sequences, most_likely_state_sequence)




#***************************  4. Learning Parameters (Baum Welch Algorithm) #***************************

print('Sampling of 10000 observations is in progress ....')
state_sequences, observation_sequences = q1.sample(10000, 20)

print('count of observations :', len(observation_sequences))
print('lenght of observations :', len(observation_sequences[0]))


# a

trueT , trueB = q4.get_TB()
#Initialising T
T = q4.create_T([0.2, 0.2, 0.2, 0.2, 0.2])
predictedT, Tscores, = q4.baum_welch(T, trueB, observation_sequences)


# b

#Initialising T
T = q4.create_T([0.2, 0.2, 0.2, 0.2, 0,2])
#Initialising B
B = np.ones((225, 16)) * (1/16)
predictedT, predictedB, Tscores, Bscores = q4.baum_welch_for_TB(T, B, observation_sequences)
Tscores = [float(x) for x in Tscores]
Bscores = [float(x) for x in Bscores]
print('T Scores are',Tscores)
print('B Scores are',Bscores)
plt.plot(Tscores)
plt.plot(Bscores)



# c
#Initialising T
T = q4.create_T([0.2, 0.2, 0.2, 0.2, 0,2])
#Initialising individual sensor B's
sensor_B = np.zeros((5, 225))
for x in range (15):
  for y in range (15):
    s=hf.get_state_number([x,y])
    if(x<=9 and y<=9):
      sensor_B[1][s]=1
    if(x>=7 and y<=9):
      sensor_B[4][s]=1
    if(x<=9 and y>=7):
      sensor_B[2][s]=1
    if(x>=7 and y>=7):
      sensor_B[3][s]=1


B = np.ones((225, 16))
for j in range(16):
  s = hf.get_observation(j) # s = [0,0,0,0]
  for i, k in enumerate(s):
    B[:,j] *= (k*sensor_B[i] + (1-k)*(1-sensor_B[i]))


# code not working properly below this
# alpha, beta, gamma are beomcing NULL matrices    
predictedT, predictedB,predicted_sensor_B, Tscores, Bscores = q4.baum_welch_for_TB2(T, sensor_B,  B, observation_sequences)




