import tensorflow as tf
import numpy as np
import gym
from tensorflow.keras.layers import Dense, Concatenate, LSTM
from tensorflow.keras import Model
from collections import deque
from rl_tf2.agents.utils import soft_update_weights, print_env_step_info

num_episodes = 100  # M
num_steps = 1000  # T
# TODO: initialize env
env_name = 'Pendulum-v0'
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_ub = env.action_space.high
action_lb = env.action_space.low

buffer_size = 10000  # replay_buffer size
lstm_size = 256
dense_size = 256
seed = 1
batch_size = 50
critic_lr = 0.0001
actor_lr = 0.0001
target_network_update_rate = 0.005
discount = 0.99
noise_std = 0.1


class RNNCritic(Model):
    """
    Input:
    obs_history - (batch, T, obs_dim)
    action_history - (batch, T, action_dim)

    Output:
    value_sequence - (batch, T)
    """
    def __init__(self, lstm_size, dense_size, name='RNNCritic'):
        super(RNNCritic, self).__init__(name=name)

        self.lstm1 = LSTM(lstm_size, name="LSTM1", return_sequences=True)
        self.dense1 = Dense(dense_size, name="HiddenDense")
        self.dense2 = Dense(1, name="OutputDense")
        self.concat = Concatenate()

    def call(self, obs_history, action_history):
        lstm_in = self.concat([obs_history, action_history])
        lstm_out = self.lstm1(lstm_in)
        x = self.dense1(lstm_out)
        x = self.dense2(x)

        # squeeze the second dimension so that the output shape will be (batch, )
        return tf.squeeze(x, axis=2)


class RNNActor(Model):
    """
    Input:
    obs_history - (batch, T, obs_dim)
    action_history - (batch, T - 1, action_dim)

    Output:
    action - (batch, T, action_dim)
    """
    def __init__(self,
                 action_dim,
                 lstm_size,
                 dense_size,
                 action_lb=None,
                 action_ub=None,
                 name='RNNActor'):
        super(RNNActor, self).__init__(name=name)

        self.action_lb = action_lb
        self.action_ub = action_ub
        self.action_dim = action_dim
        self.lstm1 = LSTM(lstm_size, name="LSTM1", return_sequences=True)
        self.dense1 = Dense(dense_size, name="HiddenDense")
        self.dense2 = Dense(action_dim, name="OutputDense")
        self.concat = Concatenate()

    def call(self, obs_history, action_history):
        batch_size = action_history.shape[0]
        action0 = tf.zeros([batch_size, 1, self.action_dim])
        print(f"obs_hist shape: {obs_history.shape}")
        print(f"action_hist shape: {action_history.shape}")
        print(f"action0 shape: {action0.shape}")
        augmented_action_history = tf.concat([action0, action_history], axis=1)

        lstm_in = self.concat([obs_history, augmented_action_history])
        x = self.lstm1(lstm_in)

        x = self.dense1(x)
        action = self.dense2(x)

        if self.action_lb is not None and self.action_ub is not None:
            mid = (self.action_lb + self.action_ub) / 2
            span = (self.action_ub - self.action_lb) / 2
            action = span * tf.nn.tanh(action) + mid
        return action


class History:
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.obs_hist = tf.zeros([0, obs_dim])

        self.action_dim = action_dim
        self.action_hist = tf.zeros([0, action_dim])

        self.reward_hist = tf.zeros([0, 1])

    @staticmethod
    def _insert(hist, new_value):
        new_value = tf.expand_dims(new_value, 0)
        return tf.concat([hist, new_value], 0)

    def insert_obs(self, obs):
        self.obs_hist = self._insert(self.obs_hist, obs)

    def insert_action(self, action):
        self.action_hist = self._insert(self.action_hist, action)

    def insert_reward(self, reward):
        self.reward_hist = self._insert(self.reward_hist, reward)

    def get_action_history(self):
        return self.action_hist

    def get_obs_history(self):
        return self.obs_hist

    def get_reward_history(self):
        return self.reward_hist


class RNNReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=10000, seed=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.seed = seed
        self.buffer = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(self.seed)

    def put(self, obs_history, action_history, reward_history):
        """
        obs_history: Tensor, sequence_length (T) * obs_dim
        action_history: Tensor, sequence_length (T) * action_dim
        reward_history: Tensor, sequence_length (T) * 1
        """
        self.buffer.append(
            tf.concat([obs_history, action_history, reward_history], 1))

    def sample(self, batch_size, replacement=True):
        idx = self.rng.choice(self.size(),
                              size=batch_size,
                              replace=replacement)
        buffer_arr = np.array(self.buffer, dtype=object)
        samples = buffer_arr[idx]
        samples_tensor = tf.convert_to_tensor(samples, dtype=tf.float32)

        batch_obs_history = samples_tensor[:, :, :self.obs_dim]
        batch_action_history = samples_tensor[:, :,
                                              self.obs_dim:(self.obs_dim +
                                                            self.action_dim)]
        batch_reward_history = samples_tensor[:, :, (
            self.obs_dim + self.action_dim):(self.obs_dim + self.action_dim +
                                             1)]
        return batch_obs_history, batch_action_history, batch_reward_history

    def size(self):
        return len(self.buffer)


def generate_action_noise(action_dim,
                          action_upper_bound,
                          action_lower_bound,
                          noise_std,
                          seed=None):
    rng = np.random.default_rng(seed)
    mean = np.zeros(action_dim)
    span = action_upper_bound - action_lower_bound
    cov = np.diag(noise_std * span)
    noise = rng.multivariate_normal(mean, cov)
    return tf.convert_to_tensor(noise, dtype=tf.float32)


replay_buffer = RNNReplayBuffer(obs_dim,
                                action_dim,
                                capacity=buffer_size,
                                seed=seed)
critic_loss_fun = tf.keras.losses.MeanSquaredError()
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

critic = RNNCritic(lstm_size, dense_size)
actor = RNNActor(action_dim,
                 lstm_size,
                 dense_size,
                 action_lb=action_lb,
                 action_ub=action_ub)
target_critic = RNNCritic(lstm_size, dense_size)
target_actor = RNNActor(action_dim,
                        lstm_size,
                        dense_size,
                        action_lb=action_lb,
                        action_ub=action_ub)

# Making the weights equal
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

for episode in range(num_episodes):
    history = History(obs_dim, action_dim)
    curr_obs = env.reset()
    sequence = []
    for t in range(num_steps):
        # TODO: noise generation
        noise = generate_action_noise(action_dim,
                                      action_ub,
                                      action_lb,
                                      noise_std,
                                      seed=seed)

        if t == 0:
            action = noise
        else:
            # add dummy batch dimension.
            obs_history = tf.expand_dims(history.get_obs_history(), axis=0)
            action_history = tf.expand_dims(history.get_action_history(),
                                            axis=0)

            action_seq = actor(obs_history, action_history)
            action = action_seq[:, -1, :] + noise

        next_obs, reward, done, _ = env.step(action)

        print_env_step_info(t, next_obs, action, reward)
        sequence.append((curr_obs, action, reward))

        history.insert_obs(curr_obs)
        history.insert_action(action)
        history.insert_reward(reward)

        curr_obs = next_obs

    replay_buffer.put(history.get_obs_history(), history.get_action_history(),
                      history.get_reward_history())

    obs_history, action_history, reward_history = replay_buffer.sample(
        batch_size)

    with tf.GradientTape() as tape:
        # obs_history: 1, ..., T; action_history: 1, ..., T-1
        # target_actions: 1, ..., T;
        target_actions = target_actor(obs_history, action_history[:, :-1, :])

        # y1*, ..., yT*
        target_critic_output = target_critic(obs_history, target_actions)

        # reward_history 1, ..., T - 1, target_values 1, ..., T-1
        target_values = reward_history[:, :
                                       -1] + discount * target_critic_output[:,
                                                                             1:, :]
        # yhat 1, ..., T
        Qpredicts = critic(obs_history, action_history)

        critic_loss = critic_loss_fun(tf.stop_gradient(target_values),
                                      Qpredicts[:, :-1, :])

    critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_gradients, critic.trainable_variables))

    with tf.GradientTape() as tape:
        # action_history: 1, ..., T-1, actor_actions: 1, ..., T
        actor_actions = actor(obs_history, action_history[:, :-1, :])
        actor_loss = -tf.math.reduce_mean(critic(obs_history, actor_actions))

    actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_gradients, actor.trainable_variables))

    soft_update_weights(target_critic.variables, critic.variables,
                        target_network_update_rate)
    soft_update_weights(target_actor.variables, actor.variables,
                        target_network_update_rate)
