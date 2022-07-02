import sys
sys.path.insert(1, 'D:/long.lt20194099/Lab/Cô Bình/Delay_Aware_And_Energy/')

import random as rd
import numpy as np
import os
from data.task.config import Config
import random as rd

class Generate:
    def __init__(self, number_of_online_user = 10):
        self.no_user = number_of_online_user
        self.task_size_interval = Config.task_data_size_by_kb_interval
        self.task_computing_size_interval = Config.task_computing_size_by_megacycle_interval
        self.server_frequency_interval = Config.server_max_frequency_by_GHZ_interval
        self.time_threshold_interval = Config.time_threshold_interval
        self.no_task_by_user_interval = Config.max_number_of_task_generated_one_user_at_given_time_interval
        self.number_of_time_slot = Config.number_of_time_slot
        self.number_of_point_per_time_slot_interval = Config.number_of_point_per_time_slot_interval
    
    
    def generate_task_info(self):
        for i in range(self.number_of_time_slot):
            with open("{}_{}/datatask{}.csv".format(str("timeslot_datasets"), self.no_user, i),"w") as output:
                number_of_point_per_time_slot = rd.randint(self.number_of_point_per_time_slot_interval[0], self.number_of_point_per_time_slot_interval[1])
                no_user = self.no_user
                
                points = []
                user_list = []
                task_input_data_size_list = []
                task_computing_size_list = []
                time_threshold_list = []
                no_task_per_point = 0

                for point in range(number_of_point_per_time_slot):
                    for user in range(1, no_user + 1):
                        no_task_by_user = rd.randint(self.no_task_by_user_interval[0], self.no_task_by_user_interval[1])

                        for _ in range(no_task_by_user):
                            no_task_per_point += 1
                            user_list.append(user)
                            task_input_data_size_list.append(rd.randint(self.task_size_interval[0], self.task_size_interval[1]))
                            task_computing_size_list.append(rd.randint(self.task_computing_size_interval[0], self.task_computing_size_interval[1]))
                            time_threshold_list.append(self.time_threshold_interval[0] + np.random.rand() * (self.time_threshold_interval[1] - self.time_threshold_interval[0]))

                indexs = list(np.sort(np.random.randint(i * 300, (i + 1) * 300, no_task_per_point)))
                points = points + indexs
                

                n = len(points)
                output.write("time step,user,task data size (KB),task computing size (Megacycles),time threshold (s)\n")
                for j in range(n):
                    output.write("{},{},{},{},{}\n".format(points[j], user_list[j], task_input_data_size_list[j], task_computing_size_list[j], time_threshold_list[j]))