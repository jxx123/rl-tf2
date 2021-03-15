import tensorflow as tf
import gym
from rl_tf2.agents.ddpg.actor_network import Actor
from rl_tf2.agents.ddpg.critic_network import Critic
from rl_tf2.agents.ddpg.ddpg_agent import DDPG
import yaml

with open('config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['env'])

if config['agent'] == 'DDPG':
    critic = Critic(hidden_size=config['critic']['hidden_size'])
    actor = Actor(env.action_space.shape[0],
                  hidden_size=config['actor']['hidden_size'],
                  action_lb=env.action_space.low,
                  action_ub=env.action_space.high)
    target_critic = Critic(hidden_size=config['critic']['hidden_size'])
    target_actor = Actor(env.action_space.shape[0],
                         hidden_size=config['actor']['hidden_size'],
                         action_lb=env.action_space.low,
                         action_ub=env.action_space.high)

    # Making the weights equal
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    agent = DDPG(
        env,
        actor,
        critic,
        target_actor,
        target_critic,
        env_name=config['env'],
        replay_size=config['replay_size'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        noise_std=config['noise_std'],
        noise_type=config['noise_type'],
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        target_network_update_rate=config['target_network_update_rate'],
        discount=config['discount'],
        max_steps_per_epoch=config['max_steps_per_epoch'],
        log_weights=config['log_weights'])
    agent.train(test_after_epoch=config['test_after_epoch'],
                render=config['render'],
                print_step_info=config['print_step_info'])
