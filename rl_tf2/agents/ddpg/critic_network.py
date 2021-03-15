import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import Model


class Critic(Model):
    def __init__(self, hidden_size=(400, 300), name='Critic'):
        super(Critic, self).__init__(name=name)

        self.state_d1 = Dense(16, activation='relu')
        self.state_d2 = Dense(32, activation='relu')
        self.action_d1 = Dense(32, activation='relu')
        self.d1 = Dense(hidden_size[0], activation='relu', name="L1")
        self.d2 = Dense(hidden_size[1], activation='relu', name="L2")
        self.d3 = Dense(1, name="L3")
        self.concat = Concatenate()

    def call(self, state, action):
        state_out = self.state_d1(state)
        state_out = self.state_d2(state_out)
        action_out = self.action_d1(action)
        #  x = tf.concat((state, action), axis=1)
        x = self.concat([state_out, action_out])
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        # squeeze the second dimension so that the output shape will be (batch, )
        return tf.squeeze(x, axis=1)
