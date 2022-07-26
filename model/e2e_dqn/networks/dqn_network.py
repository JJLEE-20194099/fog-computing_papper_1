from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(1, '/content/drive/MyDrive/fog_computing/delay_aware_and_energy/')


from model.e2e_dqn.networks.config import Config

from rl.memory import SequentialMemory

class DQNNetwork(Model):
    def __init__(self, state_size, num_actions):
        super(DQNNetwork, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions
        self.memory = SequentialMemory(limit=5000, window_length=1)
    
    def build_model(self):
        input = Input(shape=(1, self.state_size))
        x = Flatten()(input)
        for i in range(Config.length_hidden_layer):
            x = Dense(Config.n_unit_in_layer[i], activation='relu')(x)
        
        output = Dense(self.num_actions, activation = 'linear')(x)
        model = Model(inputs=input, outputs=output)

        model.summary()
        return model
    
    def call(self, input):
        x = Flatten()(input)
        for i in range(Config.length_hidden_layer):
            x = Dense(Config.n_unit_in_layer[i], activation='relu')(x)
        
        output = Dense(self.num_actions, activation = 'linear')(x)
        return output
