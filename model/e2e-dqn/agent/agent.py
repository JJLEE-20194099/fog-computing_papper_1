from collections import deque
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from e2e_dqn.environment.atari_env import Environment
from e2e_dqn.networks.dqn_network import DQNNetwork
from e2e_dqn.utils.memory import ReplayMemory, Transition
from datetime import datetime


class Agent():
    def __init__(self, name="E2E-DRL"):
        self.name = env
        self.env = Environment(self.name, train=True)
        self.discount_factor = 0.9
        self.minibatch_size = 32
        self.update_frequency = 4
        self.target_network_update_freq = 1000
        self.agent_history_length = 4
        self.memory = ReplayMemory(capacity=1000, minibatch_size=self.minibatch_size)
        self.main_network = DQNNetwork(num_actions=self.env.get_action_space_size(), agent_history_length=self.agent_history_length)
        self.target_network = DQNNetwork(num_actions=self.env.get_action_space_size(), agent_history_length = self.agent_history_length)
        self.optimizer = Adam(learning_rate = 1e-4, epsilon=1e-6)
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = 1000000
        self.replay_start_size = 10000
        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")
        self.training_frames = int(1e7)
        self.log_path = "./log/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + self.name
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")
        self.life_time = None
        self.print_log_interval = 10
        self.save_weight_interval = 10

        self.env.reset()
        _, _, _, info = self.env.step(0)
        if info["ale.lives"] > 0:
            self.life_game = True
        else:
            self.life_game = False
        
    def get_action(self, state, exploration_rate):
        recent_state = tf.expand_dims(state, axis = 0)
        if tf.random.uniform((), minval = 0, maxval = 1, dtype=tf.float32) < exploration_rate:
            action = tf.random.uniform((), minval = 0, maxval = self.env.get_action_space_size() dtype=tf.int32)
        else:
            q_value = self.main_network(tf.cast(recent_state, tf.float32))
            action = tf.cast(tf.squeeze(tf.math.argmax(q_value, axis=1)), dtype=tf.int32)
        
        return action
    
    def get_eps(self, current_step, terminal_eps=0.01, terminal_factor = 25):
        return 0.8
    
    def update_main_q_network(self, state_batch, reward_batch, next_state_batch, terminal_batch):
        with tf.GradientTape() as tape:
            next_state_q = self.target_network(next_state_batch)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            expected_q = reward_batch  + self.discount_factor * next_state_max_q * (1 - tf.cast(terminal_batch, tf.float32))
            main_q = tf.reduce_sum(self.main_network(state_batch) * tf.one_hot(action_batch, self.env.get_action_space_size(), 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)
        
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))
        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss
    
    def update_target_network(self):
        main_vars = self.main_network.trainable_variables
        target_vars = self.target_network.trainable_variables

        for main_var, target_var in zip(main_vars, target_vars):
            target_var.assign(main_var)
    
    def train(self):
        total_step = 0
        episode = 0
        latest_100_score = deque(maxlen = 100)

        while total_step < self.training_frames:
            state = self.env.reset()
            episode_step = 0
            episode_score = 0.0
            done = False

            while not done:
                eps = self.get_eps(tf.constant(total_step, tf.float32))
                action = self.get_action(tf.constant(state), tf.constant(eps, tf.float32))

                next_state, reward, done, info = self.env.step(action)
                episode_score += reward
                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                if total_step % self.update_frequency == 0 and total_step > self.replay_start_size:
                    indicies = self.memory.get_minibatch_indices()
                    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.generate_minibatch_samples(indicies)
                    self.update_main_q_network(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                
                if total_step % self.target_network_update_freq ==0 and total_step > self.replay_start_size:   
                    loss = self.update_target_network()
                
                total_step += 1
                eposide_step += 1

                if done:
                    lastest_100_score.append(episode_score)
                    self.write_summary(episode, lastest_100_score, episode_score, total_step, eps)
                    episode += 1
                    
                    if episode % self.print_log_interval== 0:
                        print("Episode: ", episode)
                        print('Lastest 100 avg: {:.4f}'.format(np.mean(lastest_100_score)))
                        print("Progress: {} / {} (:.2f) % ").format(total_step, self.training_frames, np.round(total_step / self.training_frames, 3))
                    
                    if eposide % self.save_weight_interval == 0:
                        print("Saving weights ...")
                        self.main_network.save_weights(self.log_path + "/weights/eposide_{}".format(eposide))
                        self.play(self.log_path + "/weights/", eposide=eposide)


        