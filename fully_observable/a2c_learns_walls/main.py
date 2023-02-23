from environments import *
import numpy as np
#from IPython.display import clear_output
from  tqdm import trange
import matplotlib.pyplot as plt
import os
from models import *
import json
import random
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
LOAD_FROM_MEMORY = True



def get_env(n=1000):
    e = Walls9x9SnakeEnvironment(n)
    e.FRUIT_REWARD = 5.
    return e
env_ = get_env()
GAMMA = .99
ITERATIONS = 12500 * 2
EPSILON = 0.1
LAMBDA_VALUE = 0.
LAMBDA_AGENT = 0.
#  ALPHA = 0.1
MODELS_PREFIX = f"models/{type(env_).__name__}/{env_.board_size}x{env_.board_size}"
os.makedirs(MODELS_PREFIX, exist_ok=True)

agent, value, avg_rewards = load_models(env_, folder_name=MODELS_PREFIX if LOAD_FROM_MEMORY else None)
avg_td_error = []
optimizer_value = tf.keras.optimizers.legacy.Adam(1e-4)
optimizer_agent = tf.keras.optimizers.legacy.Adam(1e-5)
eligibility_trace_agent = [tf.zeros_like(layer) for layer in agent.trainable_weights]
eligibility_trace_value = [tf.zeros_like(layer) for layer in value.trainable_weights]


plt.ion()
fig, axs = plt.subplots(1,2, figsize=(10,3))
fig_anim, ax_anim = plt.subplots(1,1, figsize=(5,5))

next_reset = 2
for iteration in trange(ITERATIONS):
    state = env_.to_state()
    with tf.GradientTape(persistent=True) as tape, tf.device("/GPU:0"):
        # PI(a|s)
        probs = agent(state)
        probs = tf.linalg.normalize(probs + EPSILON, ord=1, axis=-1)[0]
        # a ~ PI(a|s)
        actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)
        # r ~ p(s,r|s,a)
        rewards = env_.move(actions)
        new_state = tf.constant(env_.to_state())

        value_state = value(state)
        new_value_state = value(new_state)
        # calculate advantage function
        td_error = tf.stop_gradient((rewards + GAMMA * new_value_state) - value_state) * -1

        # gather sampled actions and scaled by the advantage (equivalent to sparse categorical crossentropy)
        actions_indexes = tf.concat((tf.range(actions.shape[0])[..., None], actions), axis=-1)
        loss_agent = tf.math.log(tf.gather_nd(probs, actions_indexes))[..., None] * tf.stop_gradient(td_error)
        loss_value = tf.stop_gradient(td_error) * value_state

        loss_agent = tf.reduce_mean(loss_agent)
        loss_value = tf.reduce_mean(loss_value)

    avg_td_error.append(tf.reduce_mean(td_error))
    avg_rewards.append(np.mean(rewards))

    gradient_agent = tape.gradient(loss_agent, agent.trainable_weights)
    gradient_value = tape.gradient(loss_value, value.trainable_weights)

    if LAMBDA_AGENT > 1e-4 or LAMBDA_VALUE > 1e-4:
        eligibility_trace_agent = [GAMMA * LAMBDA_AGENT * layer + gradient for layer, gradient in
                                   zip(eligibility_trace_agent, gradient_agent)]
        eligibility_trace_value = [GAMMA * LAMBDA_VALUE * layer + gradient for layer, gradient in
                                   zip(eligibility_trace_value, gradient_value)]
        optimizer_value.apply_gradients(zip(eligibility_trace_value, value.trainable_weights))
        optimizer_agent.apply_gradients(zip(eligibility_trace_agent, agent.trainable_weights))
    else:
        optimizer_value.apply_gradients(zip(gradient_value, value.trainable_weights))
        optimizer_agent.apply_gradients(zip(gradient_agent, agent.trainable_weights))

    # every N iterations, plot the avg reward of each chunk of 100 iterations, and save the models
    N = 100
    if len(avg_rewards) % N == 0 and len(avg_rewards) > 0:
        # save
        with open(f"{MODELS_PREFIX}/training.txt", "w+") as file:
            json.dump(np.array(avg_rewards).tolist(), file)
        agent.save_weights(f"{MODELS_PREFIX}/agent")
        value.save_weights(f"{MODELS_PREFIX}/value")
