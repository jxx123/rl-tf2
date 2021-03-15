import tensorflow as tf

#  @tf.function
#  def soft_update_weights(target_network, network, update_rate):
#      for i, weights in enumerate(network.weights):
#          updated_weights = update_rate * weights + (
#              1 - update_rate) * target_network.weights[i]
#          target_network.weights[i].assign(updated_weights)


@tf.function
def soft_update_weights(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def print_env_step_info(step, obs, action, reward):
    print(
        f'Step {step} - Observation: {obs}, Action: {action}, Reward: {reward}'
    )
