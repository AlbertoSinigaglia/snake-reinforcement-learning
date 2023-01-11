from environments import *
import numpy as np
from tqdm import trange
from models import *
import os
import json
import random

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
LOAD_FROM_MEMORY = False


def get_env(n=1000):
    return OriginalSnakeEnvironment(n, 15)


env_ = get_env()
# reward discount
GAMMA = .9
# number of iterations for the training (snake steps)
ITERATIONS = 100000
# Eps to add to the policy probabilities to ensure a bit of exploration
EPSILON = 0.1
# lambda parameters for value and policy wrt eligibility traces
LAMBDA_VALUE = 0.1
LAMBDA_AGENT = 0.1

# percentage of boards to sample each time for the gradient estimation
# used to de-correlate gradients

MODELS_PREFIX = f"py/models/{type(env_).__name__}/{env_.board_size}x{env_.board_size}"
os.makedirs(MODELS_PREFIX, exist_ok=True)
agent, value, avg_rewards = load_models(env_, folder_name=MODELS_PREFIX if LOAD_FROM_MEMORY else None)
avg_td_error = []
agent, value, avg_rewards = load_models(env_, folder_name=MODELS_PREFIX if LOAD_FROM_MEMORY else None)
optimizer_value = K.optimizers.Adam(1e-5)
optimizer_agent = K.optimizers.Adam(1e-6)
eligibility_trace_agent = [tf.zeros_like(layer) for layer in agent.trainable_weights]
eligibility_trace_value = [tf.zeros_like(layer) for layer in value.trainable_weights]
for _ in trange(ITERATIONS):
    state = env_.to_state()
    with tf.GradientTape(persistent=True) as tape:

        original_probs = agent(state)
        actions = tf.random.categorical(tf.math.log(tf.stop_gradient(original_probs)), 1, dtype=tf.int32)

        rewards = env_.move(actions)
        new_state = tf.constant(env_.to_state())

        value_state = value(state)
        new_value_state = value(new_state)
        td_error = tf.stop_gradient((rewards + GAMMA * new_value_state) - value_state) * -1

        actions_indexes = tf.concat((tf.range(actions.shape[0])[..., None], actions), axis=-1)
        loss_agent = tf.stop_gradient(td_error) * \
                     tf.math.log(1e-10 + tf.gather_nd(original_probs, actions_indexes))[..., None]
        loss_value = tf.stop_gradient(td_error) * value_state

        loss_agent = tf.reduce_mean(loss_agent)
        loss_value = tf.reduce_mean(loss_value)

    avg_td_error.append(tf.reduce_mean(td_error))
    avg_rewards.append(np.mean(rewards == env_.FRUIT_REWARD))

    if LAMBDA_AGENT > 1e-4 or LAMBDA_VALUE > 1e-4:
        gradient_agent = tape.gradient(loss_agent, agent.trainable_weights)
        gradient_value = tape.gradient(loss_value, value.trainable_weights)
        eligibility_trace_agent = [GAMMA * LAMBDA_AGENT * layer + gradient for layer, gradient in
                                   zip(eligibility_trace_agent, gradient_agent)]
        eligibility_trace_value = [GAMMA * LAMBDA_VALUE * layer + gradient for layer, gradient in
                                   zip(eligibility_trace_value, gradient_value)]
        optimizer_value.apply_gradients(zip(eligibility_trace_value, value.trainable_weights))
        optimizer_agent.apply_gradients(zip(eligibility_trace_agent, agent.trainable_weights))
        del gradient_agent
        del gradient_value
    else:
        optimizer_value.minimize(loss_value, value.trainable_weights, tape=tape)
        optimizer_agent.minimize(loss_agent, agent.trainable_weights, tape=tape)

    N = 1000
    if len(avg_rewards) % N == 0 and len(avg_rewards) > 0:
        points_r = np.array(avg_rewards).reshape((-1, N)).mean(axis=-1)
        # save
        with open(f"{MODELS_PREFIX}/training.txt", "w+") as file:
            json.dump(np.array(avg_rewards).tolist(), file)
        agent.save_weights(f"{MODELS_PREFIX}/agent")
        value.save_weights(f"{MODELS_PREFIX}/value")
