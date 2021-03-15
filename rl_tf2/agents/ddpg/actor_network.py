import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class Actor(Model):
    def __init__(self,
                 action_dim,
                 action_lb=None,
                 action_ub=None,
                 hidden_size=(400, 300),
                 name='Actor'):
        super(Actor, self).__init__(name=name)
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.action_dim = action_dim
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.d1 = Dense(hidden_size[0], activation='relu', name="L1")
        self.d2 = Dense(hidden_size[1], activation='relu', name="L2")
        self.d3 = Dense(action_dim, name="L3", kernel_initializer=last_init)

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        action = self.d3(x)
        if self.action_lb is not None and self.action_ub is not None:
            mid = (self.action_lb + self.action_ub) / 2
            span = (self.action_ub - self.action_lb) / 2
            action = span * tf.nn.tanh(action) + mid
        return action
