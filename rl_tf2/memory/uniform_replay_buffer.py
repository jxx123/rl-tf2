from collections import deque
import numpy as np
import tensorflow as tf
from rl_tf2.agents.utils import print_env_step_info


class UniformReplayBuffer:
    def __init__(self, capacity=10000, seed=None):
        self.capacity = capacity
        self.seed = seed
        self.buffer = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(self.seed)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size, replacement=True):
        idx = self.rng.choice(self.size(),
                              size=batch_size,
                              replace=replacement)
        buffer_arr = np.array(self.buffer, dtype=object)
        samples = buffer_arr[idx]

        # asfarray is able to convert bool to float in dones
        states, actions, rewards, next_states, dones = tuple(
            map(
                lambda x: tf.convert_to_tensor(np.asfarray(x),
                                               dtype=tf.float32),
                zip(*samples)))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


if __name__ == '__main__':
    import gym
    env = gym.make('MountainCarContinuous-v0')

    replay_buffer = UniformReplayBuffer()
    state = env.reset()
    for step in range(100):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print_env_step_info(step, next_state, action, reward)
        replay_buffer.put(state, action, reward, next_state, done)
        state = next_state

    states, actions, rewards, next_states, dones = replay_buffer.sample(32)
    #  print(replay_buffer.buffer)
    print(states)
