from collections import namedtuple
import random
from logger import logger
import tensorflow as tf
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))

class ReplayMemory:
    def __init__(self, capacity, state_shape=(84, 84), history_length=4, minibatch_size=32, verbose=True):
        self.capacity = int(capacity)
        self.history_length = history_length
        self.minibatch_size = minibatch_size
        self._memory = []
        self._index = 0
        self._full = False
        self.verbose = verbose
    
        if verbose:
            self.total_est_mem = self.capacity * (np.prod(state_shape) * 4 * 2 + 4 + 4 + 1) / 1024 ** 3
            logger.info("Estimated memory useage ONLY for storing replays: {:.4f} GB".format(self.total_est_mem))
    
    def __len__(self):
        return len(self._memory)
    
    def __getitem__(self, key):
        return self._memory[key]
    
    def __repr__(self):
        return self.__class__.__name___ + "({})".format(self.total_est_mem)
    
    def cur_index(self):
        return self._index
    
    def is_full(self):
        return self._full
    
    def push(self, state, action, reward, next_state, terminal):
        trsn = Transition(state, action, reward, next_state, terminal)
        if len(self._memory) < self.capacity:
            self._memory.append(None)
        
        if self._index + 1 == self.capacity:
            self._full = True
            if self.verbose:
                logger.info("Replay memory is full")
        
        self._memory[self._index] = trsn
        self._index = (self._index + 1) % self.capacity
    
    def get_minibatch_indicies(self):
        indicies = []
        while len(indicies) < self.minibatch_size:
            while True:
                if self.is_full():
                    index = np.random.randint(low=self.history_length, high=self.capacity, dtype=np.int32)
                else:
                    index = np.random.randint(low=self.history_length, high=self._index, dtype=np.int32)
                
                if np.any([sample.terminal for sample in self._memory[index - self.history_length:index]]):
                    continue
                indicies.append(index)
                break
        
        return indicies
    
    def generate_minibatch_samples(self, indicies):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = [], [], [], [], []

        for index in indicies:
            selected_mem = self._memory[index]
            state_batch.append(tf.constant(selected_mem.state, tf.float32))
            action_batch.append(tf.constant(selected_mem.action, tf.float32))
            reward_batch.append(tf.constant(selected_mem.reward, tf.float32))
            next_state_batch.append(tf.constant(selected_mem.next_state, tf.float32))
            terminal_batch.append(tf.constant(selected_mem.terminal, tf.float32))

        return tf.stack(state_batch, axis=0), tf.stack(action_batch, axis=0), tf.stack(reward_batch, axis=0), tf.stack(next_state_batch, axis=0), tf.stack(terminal_batch, axis=0)
        