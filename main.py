from turtle import window_width
import numpy as np

from model.e2e_dqn.environment.env import Environment
from model.e2e_dqn.networks.dqn_network import DQNNetwork
from model.e2e_dqn.agent.agent import Agent

from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout, Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

np.random.seed(123)
agent = Agent()
agent.train()



