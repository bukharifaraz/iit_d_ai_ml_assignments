import grid as gd 
import matplotlib.pyplot as plt
import numpy as np
import helperFunctions as hf

#returns transition matrix based on p=[0.4, 0.3, 0.1, 0.1, 0.1]
#returns emission matrix based on sensor observation probabilities (given in question)
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


def forward_inference (obs, T, B):
  #alpha
  alpha = np.zeros((225, len(obs)))
  #filling first column
  alpha[0][0]=1*B[0][hf.get_observation_number(obs[0])]
  #filling remaining columns
  nghbr = hf.get_neighbours()
  for o_number in range (1, len(obs)):
    o=hf.get_observation_number(obs[o_number])
    for s in range(225):
      op = B[s][o]
      n=nghbr[s]
      alpha[s, o_number] = np.sum(alpha[n, o_number-1] * T[n, s]) * B[s, o]
  return alpha


def backward_inference(obs, T, B):
  #beta
  beta = np.zeros((225, len(obs))) 
  #filling last column
  beta[:, -1] = 1
  #filling remaining columns
  nghbr = hf.get_neighbours()
  for o_number in range (len(obs)-2, -1, -1):
    o=hf.get_observation_number(obs[o_number+1])
    for s in range(0, 225):
      # Get the neighbors for state s
      n=nghbr[s]
      # Compute the transition and observation probabilities
      T_s_n = T[s, n]
      B_s_n = B[n, o]
      # Update beta values
      beta[s, o_number] = np.sum(T_s_n * B_s_n * beta[n, o_number + 1])
  return beta


# E-Step: computing gamma, ksi
def e_step(alpha, beta, obs, T, B):
  num_states = 225
  num_obs = len(obs)
  # Building gamma table
  gamma = alpha * beta
  gamma /= np.sum(gamma, axis=0)

  # Precompute the observation indices
  obs_indices = np.array([hf.get_observation_number(o) for o in obs[1:]])
  # Building ksi table
  ksi = np.zeros((num_obs-1, num_states, num_states))
  # Vectorized computation of ksi
  for t in range(num_obs-1):
    alpha_t = alpha[:, t].reshape(-1, 1)  # Shape (num_states, 1)
    beta_t1 = beta[:, t+1].reshape(1, -1)  # Shape (1, num_states)
    B_t1 = B[:, obs_indices[t]].reshape(1, -1)  # Shape (1, num_states)
    numerator = alpha_t * T * B_t1 * beta_t1
    denominator = np.sum(numerator)
    ksi[t] = numerator / denominator

  return gamma, ksi


#-----------------------------------------------------
# Functions for M-Step
#-----------------------------------------------------


def get_p_denominator(ksi):
  denominator = 0
  num_states = 225
  time_steps = 19

    # Precompute the state numbers for all [x, y] pairs
  state_matrix = np.array([[hf.get_state_number([x, y]) for x in range(1, 16)] for y in range(1, 16)])

    # Calculate the sum of ksi[t][s][s] for all time steps and states
  for t in range(time_steps):
    for y in range(15):
      for x in range(15):
        s = state_matrix[y, x]
        # Add ksi[t][s][s]
        denominator += ksi[t, s, s]
        # Add ksi[t][s][neighbor_state] for each valid neighbor
        if x > 0:  # Left neighbor
          denominator += ksi[t, s, state_matrix[y, x - 1]]
        if x < 14:  # Right neighbor
          denominator += ksi[t, s, state_matrix[y, x + 1]]
        if y > 0:  # Top neighbor
          denominator += ksi[t, s, state_matrix[y - 1, x]]
        if y < 14:  # Bottom neighbor
          denominator += ksi[t, s, state_matrix[y + 1, x]]

  return denominator


#-----------------------------------------------------
# Baum Welch
#-----------------------------------------------------

#receives initialized T, initialized B, and observations
def baum_welch(T, B, observation_sequences):
  #True T, B
  T_true , B_true = get_TB()
  scores = []
  it=1
  for iteration in range(20):
    print('Iternation Number: ', it)
    p_numerator = [0,0,0,0,0] #right, up, down, left, same
    #denom=0
    p=[0,0,0,0,0]
    j=1
    for observation in observation_sequences[:]:
      print('-- Running on observation number ',j, '... ')

      #E-Step
      alpha = forward_inference(observation, T, B)
      beta  = backward_inference(observation, T, B)
      gamma, ksi = e_step(alpha, beta, observation, T, B)

      # M-Step
      
      #p_right
      indices = np.arange(209)
      p_numerator[0] = p_numerator[0] + np.sum(ksi[:, indices, indices + 15]) #get_pr(ksi)
      #p_left
      indices = np.arange(15, 225)
      p_numerator[3] = p_numerator[3] + np.sum(ksi[:, indices, indices - 15])#get_pl(ksi)
      #p_up
      indices = np.arange(225)
      indices = indices[indices % 15 != 14]
      p_numerator[1] = p_numerator[1] + np.sum(ksi[:, indices, indices + 1])#get_pu(ksi)
      #p_down
      indices = np.arange(225)
      indices = indices[indices % 15 != 0]
      p_numerator[2] = p_numerator[2] + np.sum(ksi[:, indices, indices - 1]) #get_pd(ksi)
      #p_same
      indices = np.arange(225)
      p_numerator[4] = p_numerator[4] + np.sum(ksi[:, indices, indices ]) #get_ps(ksi)
      
      j=j+1 #next observation
      
    print('\n After ',it, ' iteration, values are:')
    prob_sum=0
    denom = np.sum(p_numerator[:5])+ 1e-10
    p[:5] = np.divide(p_numerator[:5], denom)
    p = [float(x) for x in p]
    print('[PR, PU, PD, PL, PS] =', p)

    newT = create_T(p)

    #calculate Divergence between T_true & newT
    score=ks_score(T_true, newT)
    print('\nScore for this iteration is:', score)
    scores.append(score)

    T = newT
    it=it+1 #next iteration
   
  print('Scores are ',scores)
  return newT, scores

#part b

def baum_welch_for_TB(T, B, observation_sequences):
  #True T, B
  T_true , B_true = get_TB()
  scoresT = []
  scoresB = []
  it=1
  binary_to_decimal = {tuple([int(b) for b in format(i, '04b')]): i for i in range(16)}
  for iteration in range(20):
    print('Iternation Number: ', it)
    p_numerator = [0,0,0,0,0] #right, up, down, left, same
    p=[0,0,0,0,0]
    expected_occurence_of_s=np.zeros((225))
    expected_occurence_of_s_with_observation_o = np.zeros((225,16))
    op_numerator = 0
    j=1
    for observation in observation_sequences[0:10000]:
      print('-- Running on observation number ',j, '... ')

      #E-Step
      alpha = forward_inference(observation, T, B)
      beta  = backward_inference(observation, T, B)
      gamma, ksi = e_step(alpha, beta, observation, T, B)
      #print('validating gamma ')
      #validate_gamma(gamma)
      #print('validating ksi ')
      #validate_ksi(ksi, gamma)
      #return ksi

      # M-Step for T
      
      #p_right
      indices = np.arange(209)
      p_numerator[0] = p_numerator[0] + np.sum(ksi[:, indices, indices + 15]) #get_pr(ksi)
      #p_left
      indices = np.arange(15, 225)
      p_numerator[3] = p_numerator[3] + np.sum(ksi[:, indices, indices - 15])#get_pl(ksi)
      #p_up
      indices = np.arange(225)
      indices = indices[indices % 15 != 14]
      p_numerator[1] = p_numerator[1] + np.sum(ksi[:, indices, indices + 1])#get_pu(ksi)
      #p_down
      indices = np.arange(225)
      indices = indices[indices % 15 != 0]
      p_numerator[2] = p_numerator[2] + np.sum(ksi[:, indices, indices - 1]) #get_pd(ksi)
      #p_same
      indices = np.arange(225)
      p_numerator[4] = p_numerator[4] + np.sum(ksi[:, indices, indices ]) #get_ps(ksi)
      
      j=j+1 #next observation
      
      # M Step for B
      np.add.at(expected_occurence_of_s, slice(None), np.sum(gamma, axis=1))
      for s in range(225): 
        for t in range(20):
          expected_occurence_of_s_with_observation_o[s][binary_to_decimal[tuple(observation[t])]]=expected_occurence_of_s_with_observation_o[s][binary_to_decimal[tuple(observation[t])]]+gamma[s][t]
      
      
    
    #updating Transition Probabilities
    denom = np.sum(p_numerator[:5])+ 1e-10
    p[:5] = np.divide(p_numerator[:5], denom)
    
    #updating Emission Probabilities
    newB=np.zeros((225,16))
    for s in range(225):
      for o in range(16):
        newB[s][o]= expected_occurence_of_s_with_observation_o[s][o]/expected_occurence_of_s[s]
    newB =  np.nan_to_num(newB, nan=0)
    
    print('\n After ',it, ' iteration, transition probabilities are:')
    p = [float(x) for x in p]
    print('[PR, PU, PD, PL, PS] =', p)

    newT = create_T(p)

    #calculate Divergence between T_true & newT
    score=ks_score(T_true, newT)
    print('\nScore for this iteration is:', score)
    scoresT.append(score)
    
    #calculate Divergence between B_true & newB
    score=ks_score(B_true, newB)
    print('\nScore for this iteration is:', score)
    scoresB.append(score)

    T = newT
    B = newB
    it=it+1 #next iteration
   
  print('Divergence Scores for T are ',scoresT)
  print('Divergence Scores for B are ',scoresB)
  return T, B, scoresT, scoresB





#part c:

def maskB(sensor_B):
  binary_to_decimal = {tuple([int(b) for b in format(i, '04b')]): i for i in range(16)}
  B = np.zeros((225, 16))
  for s in range(225):
    for o in range(16):
      obs = hf.get_observation(o)
      B[s][o]=sensor_B[1][s][obs[0]]*sensor_B[2][s][obs[1]]*sensor_B[3][s][obs[2]]*sensor_B[4][s][obs[3]]
  return B 
    


def baum_welch_for_TB2(T, sensor_B, B, observation_sequences):
  #True T, B
  T_true , B_true = get_TB()
  scoresT = []
  scoresB = []
  it=1
  binary_to_decimal = {tuple([int(b) for b in format(i, '04b')]): i for i in range(16)}
  for iteration in range(2):
    print('Iternation Number: ', it)
    p_numerator = [0,0,0,0,0] #right, up, down, left, same
    p=[0,0,0,0,0]
    expected_occurence_of_s=np.zeros((225))
    expected_occurence_of_s_with_observation_o = np.zeros((5, 225,2))
    op_numerator = 0
    j=1
    for observation in observation_sequences[0:5]:
      print('-- Running on observation number ',j, '... ')
#      observation = observation_sequences[0]
      #E-Step
      alpha = forward_inference(observation, T, B)
      beta  = backward_inference(observation, T, B)
      gamma, ksi = e_step(alpha, beta, observation, T, B)
      #validate_alpha_beta(alpha, beta)
      #print('validating gamma ')
      #validate_gamma(gamma)
      #print('validating ksi ')
      #validate_ksi(ksi, gamma)
      #return ksi

      # M-Step for T
      
      #p_right
      indices = np.arange(209)
      p_numerator[0] = p_numerator[0] + np.sum(ksi[:, indices, indices + 15]) #get_pr(ksi)
      #p_left
      indices = np.arange(15, 225)
      p_numerator[3] = p_numerator[3] + np.sum(ksi[:, indices, indices - 15])#get_pl(ksi)
      #p_up
      indices = np.arange(225)
      indices = indices[indices % 15 != 14]
      p_numerator[1] = p_numerator[1] + np.sum(ksi[:, indices, indices + 1])#get_pu(ksi)
      #p_down
      indices = np.arange(225)
      indices = indices[indices % 15 != 0]
      p_numerator[2] = p_numerator[2] + np.sum(ksi[:, indices, indices - 1]) #get_pd(ksi)
      #p_same
      indices = np.arange(225)
      p_numerator[4] = p_numerator[4] + np.sum(ksi[:, indices, indices ]) #get_ps(ksi)
      
      j=j+1 #next observation
      
      # M Step for sensor 1
      np.add.at(expected_occurence_of_s, slice(None), np.sum(gamma, axis=1))
      for sensor in range(1,5):
        for s in range(225): 
          for t in range(20):
            expected_occurence_of_s_with_observation_o[sensor][s][observation[t][0]]=expected_occurence_of_s_with_observation_o[sensor][s][observation[t][0]]+gamma[s][t]
      
      
    
    #updating Transition Probabilities
    denom = np.sum(p_numerator[:5])+ 1e-10
    p[:5] = np.divide(p_numerator[:5], denom)
    
    #updating Emission Probabilities of sensor-B
    new_sensor_B = np.zeros((5, 225, 2))
    for sensor in range(1,5):
      for s in range(225):
        for o in range(2):
          new_sensor_B[sensor][s][o]= expected_occurence_of_s_with_observation_o[sensor][s][o]/expected_occurence_of_s[s]
    new_sensor_B =  np.nan_to_num(new_sensor_B, nan=0)
    
    
    #Masking
    newB = maskB(new_sensor_B)
    
    print('\n After ',it, ' iteration, transition probabilities are:')
    p = [float(x) for x in p]
    print('[PR, PU, PD, PL, PS] =', p)

    newT = create_T(p)

    #calculate Divergence between T_true & newT
    score=ks_score(T_true, newT)
    print('\nScore for this iteration is:', score)
    scoresT.append(score)
    
    #calculate Divergence between B_true & newB
    score=ks_score(B_true, newB)
    print('\nScore for this iteration is:', score)
    scoresB.append(score)

    T = newT
    B = newB
    sensor_B=new_sensor_B
    it=it+1 #next iteration
   
  print('Divergence Scores for T are ',scoresT)
  print('Divergence Scores for B are ',scoresB)
  return T, B, sensor_B, scoresT, scoresB






# E-Step: computing gamma, ksi
def e_step_slow(alpha, beta, obs, T, B):
  #buidling gamma table
  denominator = np.sum(alpha*beta, axis=0)
  gamma = alpha*beta
  for t in range(len(obs)):
    gamma[:,t]=gamma[:,t]/denominator[t]

  #building ksi table
  ksi = np.zeros((len(obs)-1, 225, 225))
  for t in range(len(obs)-1):
    for i in range(225):
      denominator = 0
      for j in range(225):
        denominator+= alpha[j][t]*beta[j][t]
      for j in range(225):
        numerator = alpha[i][t]*T[i][j]*B[j][hf.get_observation_number(obs[t+1])]*beta[j][t+1]
        ksi[t][i][j]=numerator / denominator
        #if(t==0 and i==0):
          #print('j is', j, 'numerator is ', numerator, 'denominator is ', denominator, 'values of ksi is ', ksi[t][i][j])

  return gamma, ksi


#-----------------------------------------------------
# Create Transition Matrix from [pr, pu, pd, pl, ps]
#-----------------------------------------------------

def create_T(T_porb):
  #get the probabilities
  pr, pu, pd, pl, ps = T_porb[0], T_porb[1], T_porb[2], T_porb[3], T_porb[4]
  #print('values received are:')
  #print([pr, pu, pd, pl, ps])
  T = np.zeros((225,225))
  for i in range(225):
    y = (i % 15) + 1
    x = (i // 15) + 1
    #print((x,y, i))
    #self transition
    T[i,i] = ps

    #move to right
    if(x + 1 <= 15):
      T[i,i+15] = pr
    else:
      T[i,i] = pr + T[i,i] 

    #move left
    if(x - 1 > 0):
      T[i,i-15] = pl
    else:
      T[i,i] = pl + T[i,i] 
      
    #move up
    if(y + 1 <= 15):
      T[i,i+1] = pu
    else:
      T[i,i] += pu

    #move down
    if(y - 1 > 0):
      T[i,i-1] = pd
    else:
      T[i,i] += pd
  return T






#-----------------------------------------------------
#Validation Functions
#-----------------------------------------------------

def validate_alpha_beta(alpha,beta):
  #validating alpha, beta
  p=0
  for i in range(225):
    p=p+alpha[i][19]
  print('P(O) calculated using alpha',p)

  p=0
  for i in range(225):
      p=p+(alpha[i][1]*beta[i][1])
  print('P(O) calculated using alpha*beta',p)
  

def validate_gamma(gamma):
  #checking the correctness of gamma
  ss=0
  for i in range(225):
    ss=ss+gamma[i][16]
  print('This sum must be one:', ss)

def validate_ksi(ksi, gamma):
  #checking the correctness of ksi
  t=2
  sum=0
  for i in range(225):
    for j in range(225):
      sum=sum+ksi[t][i][j]
  print('This sum must be one:', sum)

  # checking relation of gamma and ksi
  s=16
  t=18
  print('Following two numbers must be equal')
  print('Gamma',[s,t],' is:',gamma[s][t]) #gamma[s][t]
  sum=0
  for i in range(225):
    sum=sum+ksi[t][s][i]
  print('summation of ksi for state to all states at all times is',[s],' is:',sum) #gamma[s][t]
  
  
#-----------------------------------------------------
# KL Score
#-----------------------------------------------------
  
def ks_score(A, newA):
  import math
  kl=0
  #print('Shape of Matrix one :', A.shape)
  #print('Shape of Matrix two :', newA.shape)
  for i in range(len(A)):
    for j in range(len(A[0])):
      epsilon = 1e-9
      A[i][j]=A[i][j]+epsilon
      newA[i][j]=newA[i][j]+epsilon
      #print('goint to divide ',A[i][j],' by ',newA[i][j], 'i is ', i, 'j is ', j)
      kl = kl+(A[i][j]*(math.log(A[i][j]/newA[i][j])))
  avg_kl = kl/(len(A)*len(A[0]))
  return avg_kl
