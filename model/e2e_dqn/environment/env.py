import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import copy
import os
import sys

from tensorflow.python.ops.gen_math_ops import Range
sys.path.insert(1, '/drive/MyDrive/fog_computing/delay_aware_and_energy/')

from model.e2e_dqn.environment.config import Config


from data.servers.config import Config as ServerConfig
import pandas as pd

class Environment:

    def __init__(self, no_user=10):
        self.no_user = no_user
        self.number_of_servers = ServerConfig.number_of_servers
        self.freq_list = self.get_freq_list()
        self.n_tasks_in_node = [0] * ServerConfig.number_of_servers
        self.action_space = self.init_action_space(ServerConfig.number_of_servers)
        self.index_of_episode= 0
        self.data = pd.read_csv('data/task/timeslot_datasets_{}/datatask{}.csv'.format(no_user, self.index_of_episode)).to_numpy()
        
        self.queue = copy.deepcopy(self.data[self.data[:, 0] == self.data[0][0]])
        self.data = self.data[self.data[:, 0] != self.data[0][0]]   

        self.time = self.queue[0][0]
        self.time_last = self.data[-1][0]
        self.observation = self.init_observation(ServerConfig.number_of_servers, self.queue)

        self.bandwidth = Config.bandwidth
        self.rayleigh_fading = Config.rayleigh_fading
        self.path_loss = Config.path_loss
        self.signal_to_noise_ratio = Config.signal_to_noise_ratio
        self.tranmission_power = Config.tranmission_power

        # self.tranmission_rate = self.bandwidth * np.log2(1 + self.tranmission_power * self.rayleigh_fading * self.path_loss / self.signal_to_noise_ratio) / 8
        self.tranmission_rate = 2000
        self.trade_off_coef = Config.f_task_energy_trade_off_coef
        self.computing_comsumption_coef = Config.computing_comsumption_coef
        self.beta_1 = Config.beta_1
        self.beta_2 = Config.beta_2
        self.C = 1


        self.sumreward = 0
        self.nreward = 0
        self.seed()

    def get_freq_list(self):
        data = pd.read_csv("data/servers/info.csv")
        freq_list = data.iloc[:, 0]
        return freq_list

    
    def init_action_space(self, number_of_servers):
        freq_list = self.freq_list
        action_space = []
        
        for index in range(1, number_of_servers + 1):
            freq = freq_list[index - 1]
            for rate in range(1, 101):
                action_space.append((index, freq * rate / 100))
        
        return action_space

    def preprocessing(self, state):
      a = 0
      for s in state:
        a += s*s
      
      res = [s / pow(a, 1/2) for s in state]
      return np.array(res)
    
    def init_observation(self, number_of_servers, queue):
        observation_list = [0.0] * number_of_servers
        observation_list.append(queue[0][2])
        observation_list.append(queue[0][3])
        observation_list.append(queue[0][4])
        observation_list.append(0.0)
        observation_list.append(0.0)
        observation_list.append(0.0)
        observation_list.append(0.0)
        observation_list.append(0.0)
        observation_list.append(0)
        
        # return self.preprocessing(np.array(observation_list))
        return observation_list

    


    def step(self, action):
        k = action[0]
        recommended_frequency = action[1]
        
        task_computing_size = self.observation[self.number_of_servers + 1]
        delta_C_ij = task_computing_size / (recommended_frequency * 1000)
        # if( self.observation[k - 1] < 1):
        #   print("{}th server:".format(k), self.observation[k - 1])
        self.observation[k - 1] += delta_C_ij
        self.observation[self.number_of_servers + 4] = delta_C_ij
        

        task_data_size = self.observation[self.number_of_servers]
        delta_T_ij = task_data_size / self.tranmission_rate
        self.observation[self.number_of_servers + 3] = delta_T_ij
        
        

        E_T_ij = delta_T_ij * self.tranmission_power
        self.observation[self.number_of_servers + 5] = E_T_ij
        E_C_ij = self.computing_comsumption_coef * np.power(recommended_frequency, 2) * task_computing_size * np.power(10, 24)
        self.observation[self.number_of_servers + 6] = E_C_ij

        E = E_T_ij + E_C_ij
        self.observation[self.number_of_servers + 7] = E
        



        cnt = 0
        if delta_T_ij >= self.observation[k - 1]:
            time_delay = delta_T_ij +  delta_C_ij
            cnt = 1
        else:
            time_delay = self.observation[k - 1] + delta_C_ij
        
        time_threshold = self.observation[self.number_of_servers + 2]

        # print("Time:", time_delay, time_threshold, cnt, delta_T_ij, delta_C_ij, self.observation[k - 1])
        if time_delay <= time_threshold:
            F_task = 1
        else:
            F_task = 0
        self.observation[self.number_of_servers + 8] = F_task

        self.n_tasks_in_node[k - 1] += 1
        reward = (1 - self.trade_off_coef) * self.beta_1 * F_task - self.trade_off_coef * self.beta_2 * np.log2(E) + self.C

        # if (self.trade_off_coef * self.beta_2 * np.log2(E) < 0):
        # print("Part1:", (1 - self.trade_off_coef) * self.beta_1 * F_task)
        # print(self.computing_comsumption_coef, np.power(recommended_frequency, 2), task_computing_size)
        # print("Part2:", self.trade_off_coef * self.beta_2 * np.log2(E), E_T_ij, E_C_ij)


        # print("Reward:", reward)      

        if len(self.queue) != 0:
            self.queue = np.delete(self.queue, (0), axis=0)
        
        if len(self.queue) == 0 and len(self.data) != 0:
            self.queue = copy.deepcopy(self.data[self.data[:, 0] == self.data[0][0]])

            time = self.data[0][0] - self.time
            for i in range(self.number_of_servers):
                self.observation[i] = max(self.observation[i] - time, 0)
          
            
            self.time = self.data[0][0]
            self.data = self.data[self.data[:, 0] != self.data[0, 0]]
        
        
        if len(self.queue) != 0:
            self.observation[self.number_of_servers] = self.queue[0][2]
            self.observation[self.number_of_servers + 1] = self.queue[0][3]
            self.observation[self.number_of_servers + 2] = self.queue[0][4]

        done = len(self.queue) == 0 and len(self.data) == 0

        if done:
            print(self.n_tasks_in_node)
            

        self.sumreward = self.sumreward + reward
        self.nreward = self.nreward + 1
        avg_reward = self.sumreward / self.nreward


        # return self.preprocessing(self.observation), reward, done, {"info": "My info"}
        return self.observation, reward, done, {"info": "My info"}
    
    def reset(self):
        if self.index_of_episode == -1:
            self.data = pd.read_csv('data/task/timeslot_datasets_{}/datatask{}.csv'.format(self.no_user, self.index_of_episode)).to_numpy()
        
            self.queue = copy.deepcopy(self.data[self.data[:, 0] == self.data[0][0]])
            self.data = self.data[self.data[:, 0] != self.data[0][0]]   

            self.time = self.queue[0][0]
            self.time_last = self.data[-1][0]
            self.observation = self.init_observation(self.number_of_servers, self.queue)
            return self.observation
        
        self.n_tasks_in_node = [0] * self.number_of_servers

        self.index_of_episode = self.index_of_episode + 1
        self.data = pd.read_csv('data/task/timeslot_datasets_{}/datatask{}.csv'.format(self.no_user, self.index_of_episode)).to_numpy()
        self.queue = copy.deepcopy(self.data[self.data[:, 0] == self.data[0][0]])
        self.data = self.data[self.data[:, 0] != self.data[0][0]]   
        self.time = self.queue[0][0]
        # print("data", self.data[0])
        # print("Time:",  self.time)
        # print("Time Last:",  self.time_last)
        for i in range(self.number_of_servers):
            self.observation[i] = max(0, self.observation[i] - (self.time - self.time_last))
        
        self.observation[self.number_of_servers] = self.queue[0][2]
        self.observation[self.number_of_servers + 1] = self.queue[0][3]
        self.observation[self.number_of_servers + 2] = self.queue[0][4]

        self.time_last = self.data[-1][0]

        # return self.preprocessing(self.observation)
        return self.observation    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    


        
    