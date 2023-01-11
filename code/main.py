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


def get_env(n=1000):
    return OriginalSnakeEnvironment(n, 15)


def norm(grad):
    return np.sum([np.linalg.norm(np.reshape(a, (-1)), ord=1) for a in grad])


env_ = get_env()
LOAD_FROM_MEMORY = False
GAMMA = .9
ITERATIONS = 100000
EPSILON = 0.1
LAMBDA_VALUE = 0.
LAMBDA_AGENT = 0.
ALPHA = 0.1


MODELS_PREFIX = f"py/models/{type(env_).__name__}/{env_.board_size}x{env_.board_size}"
os.makedirs(MODELS_PREFIX, exist_ok=True)
agent, value, avg_rewards = load_models(env_, folder_name=MODELS_PREFIX if LOAD_FROM_MEMORY else None)
avg_td_error = []

optimizer_value = K.optimizers.Adam(1e-4)
optimizer_agent = K.optimizers.Adam(1e-5)

eligibility_trace_agent = [tf.zeros_like(layer) for layer in agent.trainable_weights]
eligibility_trace_value = [tf.zeros_like(layer) for layer in value.trainable_weights]
for _ in trange(ITERATIONS):
    state = env_.to_state()
    with tf.GradientTape(persistent=True) as tape:

        original_probs = agent(state)
        mask_with = 0.
        mask = get_probabilities_mask(env_.boards, original_probs.shape, mask_with=mask_with)
        probs = tf.linalg.normalize((original_probs + EPSILON) * mask, ord=1, axis=-1)[0]

        actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)

        rewards = env_.move(actions)
        new_state = tf.constant(env_.to_state())

        value_state = value(state)
        new_value_state = value(new_state)
        td_error = tf.stop_gradient((rewards + GAMMA * new_value_state) - value_state) * -1

        actions_indexes = tf.concat((tf.range(actions.shape[0])[..., None], actions), axis=-1)
        loss_agent = tf.stop_gradient(td_error) * \
                     tf.math.log(1e-10 + tf.gather_nd(original_probs, actions_indexes))[..., None]
        loss_value = tf.stop_gradient(td_error) * value_state

        not_available_actions_indexes = tf.convert_to_tensor(
            np.argwhere(np.logical_and(mask == mask_with, original_probs > .2)),
             dtype=tf.int32
        )
        not_available_td_error = tf.abs(tf.gather_nd(td_error, not_available_actions_indexes[:,0,None]))
        loss_not_available_single = tf.gather_nd(original_probs, not_available_actions_indexes)[...,None] * not_available_td_error
        loss_not_available = tf.reduce_mean(tf.math.log(loss_not_available_single))

        loss_agent = tf.reduce_mean(loss_agent) + loss_not_available * ALPHA
        loss_value = tf.reduce_mean(loss_value)

    avg_td_error.append(tf.reduce_mean(td_error))
    avg_rewards.append(np.mean(rewards == env_.FRUIT_REWARD))

    norm_available = norm(tape.gradient(loss_agent, agent.trainable_weights))
    norm_not_available = norm(tape.gradient(loss_not_available, agent.trainable_weights))
    ratio = min(norm_available / (norm_not_available+1e-1), 1.)

    gradient_agent_actions_chosen = tape.gradient(loss_agent, agent.trainable_weights)
    gradient_agent_actions_unavailable = tape.gradient(loss_not_available, agent.trainable_weights)
    gradient_agent = [
        u*ratio + c
        for u, c in zip(gradient_agent_actions_unavailable, gradient_agent_actions_chosen)]
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

    N = 1000
    if len(avg_rewards) % N == 0 and len(avg_rewards) > 0:
        points_r = np.array(avg_rewards).reshape((-1, N)).mean(axis=-1)
        with open(f"{MODELS_PREFIX}/training.txt", "w+") as file:
            json.dump(np.array(avg_rewards).tolist(), file)
        agent.save_weights(f"{MODELS_PREFIX}/agent")
        value.save_weights(f"{MODELS_PREFIX}/value")

    del loss_agent
    del loss_value
    del loss_not_available
    #del not_available_actions_td_error
    #del not_available_actions_indexes
    del td_error
    del new_value_state
    del value_state
    del state
    del new_state
    del rewards
    del actions
    del probs
    del original_probs
    del mask
    del tape
