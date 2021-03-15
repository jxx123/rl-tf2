import tensorflow as tf
import numpy as np
import datetime
from rl_tf2.memory.uniform_replay_buffer import UniformReplayBuffer
from rl_tf2.agents.utils import soft_update_weights, print_env_step_info


class DDPG:
    def __init__(self,
                 env,
                 actor,
                 critic,
                 target_actor,
                 target_critic,
                 env_name=None,
                 epochs=100,
                 max_steps_per_epoch=1000,
                 batch_size=32,
                 replay_size=1000,
                 actor_lr=0.01,
                 critic_lr=0.01,
                 noise_std=0.1,
                 noise_type='normal',
                 discount=0.9,
                 target_network_update_rate=0.005,
                 train_log_dir=None,
                 test_log_dir=None,
                 log_weights=False,
                 seed=None):
        self.env = env
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.ou_noise = None
        self.discount = discount
        self.target_network_update_rate = target_network_update_rate
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)
        self.critic = critic
        self.actor = actor
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.replay_buffer = UniformReplayBuffer(capacity=self.replay_size,
                                                 seed=seed)

        self.critic_loss = tf.keras.losses.MeanSquaredError()
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.actor_lr)

        self.critic_loss_metric = tf.keras.metrics.Mean(name='critic_loss')
        self.actor_loss_metric = tf.keras.metrics.Mean(name='actor_loss')
        self.epoch_reward_metric = tf.keras.metrics.Sum(name='epoch_reward')

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not env_name:
            env_name = 'anonymous_env'
        if not train_log_dir:
            train_log_dir = f'.tensorboard/logs/ddpg/{env_name}/{current_time}/train'

        self.train_log_dir = train_log_dir
        if not test_log_dir:
            test_log_dir = f'.tensorboard/logs/ddpg/{env_name}/{current_time}/test'
        self.test_log_dir = test_log_dir
        self.log_weights = log_weights

        self.train_summary_writer = tf.summary.create_file_writer(
            self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(
            self.test_log_dir)

    @property
    def action_dim(self):
        return self.env.action_space.shape

    @property
    def action_upper_bound(self):
        return self.env.action_space.high

    @property
    def action_lower_bound(self):
        return self.env.action_space.low

    @tf.function
    def train_networks(self, states, actions, rewards, next_states, dones):
        """
        Wrap intensive computations into tf.function
        """
        with tf.GradientTape() as tape:
            # Compute TD targets
            target_next_actions = self.target_actor(next_states,
                                                    training=False)
            target_next_q_values = (1.0 - dones) * self.target_critic(
                next_states, target_next_actions, training=False)
            td_targets = rewards + self.discount * target_next_q_values
            Qpredicts = self.critic(states, actions, training=True)
            critic_loss = self.critic_loss(tf.stop_gradient(td_targets),
                                           Qpredicts)

        critic_gradients = tape.gradient(critic_loss,
                                         self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_actions = self.actor(states, training=True)
            actor_loss = -tf.math.reduce_mean(
                self.critic(states, actor_actions))

        actor_gradients = tape.gradient(actor_loss,
                                        self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables))

        # record the losses
        self.critic_loss_metric(critic_loss)
        self.actor_loss_metric(actor_loss)

    def train_step(self, states, actions, rewards, next_states, dones):
        # Train critic and actor networks
        self.train_networks(states, actions, rewards, next_states, dones)

        # Soft update the target networks
        soft_update_weights(self.target_critic.variables,
                            self.critic.variables,
                            self.target_network_update_rate)
        soft_update_weights(self.target_actor.variables, self.actor.variables,
                            self.target_network_update_rate)

    def choose_action(self, state, noise=False):
        # add batch dimension to state
        state_batch = tf.expand_dims(state, axis=0)
        action_batch = self.actor(state_batch, training=False)

        # remove the batch dimension
        action = tf.squeeze(action_batch, axis=0)

        if noise:
            action += self.generate_action_noise()
            action = tf.clip_by_value(action, self.action_lower_bound,
                                      self.action_upper_bound)
        return action

    def generate_action_noise(self):
        if self.noise_type == 'OU':
            if not self.ou_noise:
                self.ou_noise = OUActionNoise(
                    mean=np.zeros(1),
                    std_deviation=float(self.noise_std) * np.ones(1))
            return tf.convert_to_tensor(self.ou_noise(), dtype=tf.float32)
        elif self.noise_type == 'normal':
            mean = np.zeros(self.action_dim)
            span = self.action_upper_bound - self.action_lower_bound
            cov = np.diag(self.noise_std * span)
            noise = self.rng.multivariate_normal(mean, cov)
            return tf.convert_to_tensor(noise, dtype=tf.float32)
        else:
            raise ValueError('Unrecognized noise_type')

    def log_network_weights(self, global_step):
        with self.train_summary_writer.as_default():
            for i, weights in enumerate(self.critic.weights):
                tf.summary.histogram(f'critic_layer_{i}',
                                     weights,
                                     step=global_step)

            for i, weights in enumerate(self.target_critic.weights):
                tf.summary.histogram(f'target_critic_layer_{i}',
                                     weights,
                                     step=global_step)

            for i, weights in enumerate(self.actor.weights):
                tf.summary.histogram(f'actor_layer_{i}',
                                     weights,
                                     step=global_step)

            for i, weights in enumerate(self.target_actor.weights):
                tf.summary.histogram(f'target_actor_layer_{i}',
                                     weights,
                                     step=global_step)

    def train(self,
              test_after_epoch=None,
              render=False,
              print_step_info=False):
        global_step = 0
        for epoch in range(self.epochs):
            self.critic_loss_metric.reset_states()
            self.actor_loss_metric.reset_states()
            self.epoch_reward_metric.reset_states()

            curr_obs = self.env.reset()
            for step in range(self.max_steps_per_epoch):
                if render:
                    self.env.render(mode='human')

                action = self.choose_action(curr_obs, noise=True)
                next_obs, reward, done, _ = self.env.step(action)

                if print_step_info:
                    print_env_step_info(step, next_obs, action, reward)

                self.replay_buffer.put(curr_obs, action, reward, next_obs,
                                       done)
                if self.replay_buffer.size() > self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                        self.batch_size)
                    self.train_step(states, actions, rewards, next_states,
                                    dones)

                if self.log_weights:
                    self.log_network_weights(global_step)
                self.epoch_reward_metric(reward)

                if done:
                    break

                # DON'T forget to update observation
                curr_obs = next_obs
                global_step += 1

            with self.train_summary_writer.as_default():
                tf.summary.scalar('critic_loss',
                                  self.critic_loss_metric.result(),
                                  step=epoch)
                tf.summary.scalar('actor_loss',
                                  self.actor_loss_metric.result(),
                                  step=epoch)
                tf.summary.scalar('epoch_reward',
                                  self.epoch_reward_metric.result(),
                                  step=epoch)

            print(f'Epoch {epoch} - Train, '
                  f'Critic Loss: {self.critic_loss_metric.result()}, '
                  f'Actor Loss: {self.actor_loss_metric.result()}, '
                  f'Epoch Reward: {self.epoch_reward_metric.result()}')

            if test_after_epoch and epoch > test_after_epoch:
                test_reward_sum = self.test(render=render,
                                            print_step_info=print_step_info)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('epoch_reward',
                                      test_reward_sum,
                                      step=epoch)
                print(f'Epoch {epoch} - Test, '
                      f'Epoch Reward: {test_reward_sum}')

    def test(self, render=False, print_step_info=False):
        self.epoch_reward_metric.reset_states()

        curr_obs = self.env.reset()
        for step in range(self.max_steps_per_epoch):
            if render:
                self.env.render(mode='human')

            action = self.choose_action(curr_obs, noise=False)
            next_obs, reward, done, _ = self.env.step(action)

            if print_step_info:
                print_env_step_info(step, next_obs, action, reward)

            self.epoch_reward_metric(reward)

            if done:
                break
            curr_obs = next_obs
        return self.epoch_reward_metric.result()


class OUActionNoise:
    def __init__(self,
                 mean,
                 std_deviation,
                 theta=0.15,
                 dt=1e-2,
                 x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +
             self.std_dev * np.sqrt(self.dt) *
             np.random.normal(size=self.mean.shape))
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
