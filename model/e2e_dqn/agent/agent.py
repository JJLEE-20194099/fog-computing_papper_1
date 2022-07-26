import pandas as pd
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from gym.utils import seeding
import gym
from collections import deque
from datetime import datetime
from model.e2e_dqn.utils.memory import ReplayMemory, Transition
from model.e2e_dqn.networks.dqn_network import DQNNetwork
from model.e2e_dqn.environment.env import Environment
import sys
sys.path.insert(
    1, '/content/content/drive/MyDrive/fog_computing/delay_aware_and_energy/')


class Agent():
    def __init__(self, name="E2E-DRL"):
        self.name = name
        self.env = Environment()
        self.discount_factor = 0.9
        self.minibatch_size = 256
        self.update_frequency = 4
        self.target_network_update_freq = 1000
        self.agent_history_length = 4
        self.memory = ReplayMemory(
            capacity=2000, minibatch_size=self.minibatch_size)
        self.main_network = DQNNetwork(state_size=len(
            self.env.observation), num_actions=len(self.env.action_space))
        self.target_network = DQNNetwork(state_size=len(
            self.env.observation), num_actions=len(self.env.action_space))
        self.optimizer = Adam(learning_rate=1e-4, epsilon=1e-6)
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = 10000
        self.replay_start_size = 5000
        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")
        self.training_frames = int(1e7)
        self.log_path = "./log/" + datetime.now().strftime("%Y%m%d_%H%M%S") + \
            "_" + self.name
        self.life_time = None
        self.print_log_interval = 1
        self.save_weight_interval = 1
        self.episode_length = 100
        self.server_info = pd.read_csv(
            '../../../data/servers/info.csv').to_numpy()[0]

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_action(self, state, exploration_rate, algo='E2E_DRL', n_users=10):
        if algo == 'greedy':
            print(self.server_info)
            num_servers = len(self.server_info)
            server_queue_time = np.array(self.env.observation[:num_servers])
            min_idx = np.argmin(server_queue_time)
            recommended_frequency = self.server_info(min_idx)
            k = min_idx + 1

        else:
            recent_state = tf.expand_dims(state, axis=0)
            random_num = tf.random.uniform(
                (), minval=0, maxval=1, dtype=tf.float32)
            if random_num < exploration_rate:
                pos = tf.random.uniform((), minval=0, maxval=len(
                    self.env.action_space), dtype=tf.int32).numpy()
            else:
                q_value = self.main_network.call(
                    tf.cast(recent_state, tf.float32))
                pos = tf.cast(tf.squeeze(tf.math.argmax(
                    q_value, axis=1)), dtype=tf.int32).numpy()
            k = self.env.action_space[pos][0]
            recommended_frequency = self.env.action_space[pos][1]
        action = [k, recommended_frequency]
        return action

    def get_eps(self, current_step, terminal_eps=0.01, terminal_frame_factor=25):

        terminal_eps_frame = self.final_explr_frame * terminal_frame_factor

        if current_step < self.replay_start_size:
            eps = self.init_explr
        elif self.replay_start_size <= current_step and current_step < self.final_explr_frame:
            eps = (self.final_explr - self.init_explr) / (self.final_explr_frame -
                                                          self.replay_start_size) * (current_step - self.replay_start_size) + self.init_explr
        elif self.final_explr_frame <= current_step and current_step < terminal_eps_frame:
            eps = (terminal_eps - self.final_explr) / (terminal_eps_frame -
                                                       self.final_explr_frame) * (current_step - self.final_explr_frame) + self.final_explr
        else:
            eps = terminal_eps
        eps = tf.constant(eps)
        return eps

    def update_main_q_network(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        with tf.GradientTape() as tape:
            next_state_q = self.target_network.call(next_state_batch)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            expected_q = reward_batch + self.discount_factor * \
                next_state_max_q * (1 - tf.cast(terminal_batch, tf.float32))

            main_q = self.main_network.call(state_batch)
            temp = main_q.numpy()
            res = []
            for t in temp:
                res.append(t[np.argmax(t)])

            main_q = tf.constant(res)
            # print(main_q[:10])
            # print(reward_batch.numpy()[:10])
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        # print("loss:", loss)

        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(
            zip(clipped_gradients, self.main_network.trainable_variables))
        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss

    def update_target_network(self):
        main_vars = self.main_network.trainable_variables
        target_vars = self.target_network.trainable_variables

        for main_var, target_var in zip(main_vars, target_vars):
            target_var.assign(main_var)

    def save_episode_info(self, name, episode, score, episode_score, total_step, number_of_completed_tasks, energy_consumption, eps):
        with open(name, "a") as fin:
            fin.write("{},{},{},{},{},{},{},{},{}\n".format(episode, episode_score, np.mean(score), self.loss_metric.result(
            ), self.q_metric.result(), total_step, number_of_completed_tasks, energy_consumption, eps))
        self.loss_metric.reset_states()
        self.q_metric.reset_states()

    def train(self, algo='E2E_DRL'):

        total_step = 0
        episode = 0
        score = deque()

        if os.path.exists("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log") == False:
            os.makedirs(
                "/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log")
        if os.path.exists("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/weights") == False:
            os.makedirs(
                "/content/drive/MyDrive/fog_computing/delay_aware_and_energy/weights")

        if algo == 'E2E_DRL':

            while total_step < self.training_frames and self.env.index_of_episode < 100:
                print("Time slot:", self.env.index_of_episode)

                with open("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log/timeslot_{}.txt".format(self.env.index_of_episode), "w") as fin:
                    fin.write(
                        "Episode,Reward,Avg Reward,Loss,Average Q,Total Frames,Number of completed tasks,Energy Consumption,Epsilon\n")
                episode_step = 1
                episode_score = 0.0
                number_of_completed_tasks = 0
                energy_consumption = 0.0
                done = False
                state = self.env.observation

                while not done:
                    eps = self.get_eps(tf.constant(total_step, tf.float32))

                    action = self.get_action(tf.constant(
                        state), tf.constant(eps, tf.float32), algo='E2E_DRL')

                    next_state, reward, done, info = self.env.step(action)
                    # print(reward, next_state)

                    episode_score += reward
                    number_of_completed_tasks += next_state[-1]
                    energy_consumption += next_state[-2]

                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state

                    if total_step % self.update_frequency == 0 and total_step > self.replay_start_size:
                        indicies = self.memory.get_minibatch_indicies()

                        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.generate_minibatch_samples(
                            indicies)

                        self.update_main_q_network(
                            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

                    if total_step % self.target_network_update_freq == 0 and total_step > self.replay_start_size:
                        loss = self.update_target_network()

                    total_step += 1
                    episode_step += 1
                    if episode_step % self.episode_length == 0 and done == False:
                        score.append(episode_score)
                        # print("timeslot_{}_episode_{}.txt".format(self.env.index_of_episode, episode))

                        self.save_episode_info("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log/timeslot_{}.txt".format(
                            self.env.index_of_episode), episode, score, episode_score, total_step, number_of_completed_tasks, energy_consumption, eps.numpy())
                        if episode % self.print_log_interval == 0:
                            # print("Episode: ", episode)
                            # print('Lastest avg: {:.4f}'.format(np.mean(score)))
                            # print("Progress: {} / {} = {} ".format(total_step, self.training_frames, np.round(total_step / self.training_frames, 3)))
                            pass

                        episode += 1

                    elif done == True:
                        score.append(episode_score)
                        print("timeslot_{}_episode_{}.txt".format(
                            self.env.index_of_episode, episode))
                        self.save_episode_info("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log/timeslot_{}.txt".format(
                            self.env.index_of_episode), episode, score, episode_score, total_step, number_of_completed_tasks, energy_consumption, eps.numpy())
                        if episode % self.print_log_interval == 0:
                            # print("Episode: ", episode)
                            # print('Lastest avg: {:.4f}'.format(np.mean(score)))
                            # print("Progress: {} / {} = {} ".format(total_step, self.training_frames, np.round(total_step / self.training_frames, 3)))
                            pass

                        if episode % self.save_weight_interval == 0:
                            print("Saving weights ...")
                            self.main_network.save_weights(
                                "/content/drive/MyDrive/fog_computing/delay_aware_and_energy/weights/episode_{}.txt".format(episode))
                            # self.test("/weights/", episode=episode)
                            pass
                        episode += 1

                        state = self.env.reset()
        elif algo == 'greedy':
            while total_step < self.training_frames and self.env.index_of_episode < 100:
                print("Time slot:", self.env.index_of_episode)

                with open("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log/timeslot_{}.txt".format(self.env.index_of_episode), "w") as fin:
                    fin.write(
                        "Episode,Reward,Avg Reward,Loss,Average Q,Total Frames,Number of completed tasks,Energy Consumption,Epsilon\n")
                episode_step = 1
                episode_score = 0.0
                number_of_completed_tasks = 0
                energy_consumption = 0.0
                done = False
                state = self.env.observation

                while not done:
                    eps = self.get_eps(tf.constant(total_step, tf.float32))

                    action = self.get_action(tf.constant(
                        state), tf.constant(eps, tf.float32), algo='greedy')

                    next_state, reward, done, info = self.env.step(action)
                    # print(reward, next_state)

                    episode_score += reward
                    number_of_completed_tasks += next_state[-1]
                    energy_consumption += next_state[-2]

                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state

                    if total_step % self.update_frequency == 0 and total_step > self.replay_start_size:
                        indicies = self.memory.get_minibatch_indicies()

                        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.generate_minibatch_samples(
                            indicies)


                    total_step += 1
                    episode_step += 1
                    if episode_step % self.episode_length == 0 and done == False:
                        score.append(episode_score)
                        # print("timeslot_{}_episode_{}.txt".format(self.env.index_of_episode, episode))

                        self.save_episode_info("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log/timeslot_{}.txt".format(
                            self.env.index_of_episode), episode, score, episode_score, total_step, number_of_completed_tasks, energy_consumption, eps.numpy())
                        if episode % self.print_log_interval == 0:
                            # print("Episode: ", episode)
                            # print('Lastest avg: {:.4f}'.format(np.mean(score)))
                            # print("Progress: {} / {} = {} ".format(total_step, self.training_frames, np.round(total_step / self.training_frames, 3)))
                            pass

                        episode += 1

                    elif done == True:
                        score.append(episode_score)
                        print("timeslot_{}_episode_{}.txt".format(
                            self.env.index_of_episode, episode))
                        self.save_episode_info("/content/drive/MyDrive/fog_computing/delay_aware_and_energy/log/timeslot_{}.txt".format(
                            self.env.index_of_episode), episode, score, episode_score, total_step, number_of_completed_tasks, energy_consumption, eps.numpy())
                        if episode % self.print_log_interval == 0:
                            # print("Episode: ", episode)
                            # print('Lastest avg: {:.4f}'.format(np.mean(score)))
                            # print("Progress: {} / {} = {} ".format(total_step, self.training_frames, np.round(total_step / self.training_frames, 3)))
                            pass
                        episode += 1

                        state = self.env.reset()

    def test(self, load_dir=None, episode=None, trial=5):
        if load_dir:
            loaded_ckpt = tf.train.latest_checkpoint(load_dir)
            self.main_network.load_weights(loaded_ckpt)

        frame_set = []
        reward_set = []
        test_env = Environment()
        for _ in range(trial):

            state = test_env.reset()
            frames = []
            test_step = 0
            test_reward = 0
            done = False
            test_memory = ReplayMemory(10000, verbose=False)

            while not done:

                action = self.get_action(tf.constant(
                    state, tf.float32), tf.constant(0.0, tf.float32))

                next_state, reward, done, info = test_env.step(action)
                test_reward += reward

                test_memory.push(state, action, reward, next_state, done)
                state = next_state

                test_step += 1

                if done:
                    test_env.reset()
                    test_step = 0
                    done = False

                if len(frames) > 10000:
                    break

            reward_set.append(test_reward)
            frame_set.append(frames)

        best_score = np.max(reward_set)
        print("Best score of current network ({} trials): {}".format(
            trial, best_score))
        best_score_ind = np.argmax(reward_set)
        if episode is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("Test score", best_score, step=episode)
