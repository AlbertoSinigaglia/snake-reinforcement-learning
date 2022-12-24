from tqdm import trange
import matplotlib.pyplot as plt
from environments import *
import multiprocessing as mp


def re_normalize_possible_actions(state, probs, mask_with=np.finfo(np.float32).min):
    state = tf.argmax(state, axis=-1)
    board_size = state.shape[-1]
    indexes = tf.where(state == 2)
    mask = np.ones_like(probs)

    # [up, right, bottom, left]
    left_border = tf.where(indexes[:, 2] == 0)
    left_border = tf.concat((left_border, tf.ones_like(left_border) * 3), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, left_border, mask_with * tf.ones(left_border.shape[0]))

    right_border = tf.where(indexes[:, 2] == board_size - 1)
    right_border = tf.concat((right_border, tf.ones_like(right_border) * 1), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, right_border, mask_with * tf.ones(right_border.shape[0]))

    top_border = tf.where(indexes[:, 1] == board_size - 1)
    top_border = tf.concat((top_border, tf.ones_like(top_border) * 0), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, top_border, mask_with * tf.ones(top_border.shape[0]))

    bottom_border = tf.where(indexes[:, 1] == 0)
    bottom_border = tf.concat((bottom_border, tf.ones_like(bottom_border) * 2), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, bottom_border, mask_with * tf.ones(bottom_border.shape[0]))
    return tf.nn.softmax(probs * tf.stop_gradient(mask))


if __name__ == '__main__':
    mp.set_start_method('spawn')

    BOARD_SIZE = 4
    ITERATIONS = 10000
    # region models
    input = K.layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, 3))
    x = K.layers.Conv2D(32, (3, 3), padding="SAME", activation="linear", use_bias=False)(input)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Conv2D(32, (3, 3), padding="SAME", activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.MaxPool2D(2)(x)

    x = K.layers.Conv2D(32, (2, 2), padding="SAME", activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Conv2D(8, (2, 2), padding="SAME", activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.GlobalAvgPool2D()(x)

    encoder = K.Model(inputs=input, outputs=x)

    dec = K.layers.Reshape((1, 1, -1))(x)
    dec = K.layers.Conv2DTranspose(32, 2, activation=tf.nn.leaky_relu)(dec)
    if BOARD_SIZE > 3:
        dec = K.layers.Conv2DTranspose(32, 2, activation=tf.nn.leaky_relu)(dec)
    if BOARD_SIZE > 4:
        dec = K.layers.Conv2DTranspose(32, 2, activation=tf.nn.leaky_relu)(dec)
    dec = K.layers.Conv2DTranspose(3, 2, activation=tf.nn.softmax)(dec)
    decoder = K.Model(inputs=x, outputs=dec)

    ae = K.models.Sequential([
        encoder,
        decoder
    ])

    policy = K.layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    policy = K.layers.Dense(64, activation=tf.nn.leaky_relu)(policy)
    policy = K.layers.Dense(4, activation=tf.nn.softmax)(policy)
    agent = K.models.Model(inputs=input, outputs=policy)

    # value = K.models.Sequential()
    # value.add(K.layers.Flatten(input_shape=(BOARD_SIZE,BOARD_SIZE,4)))
    # value.add(K.layers.Dense(256, activation='selu'))
    # value.add(K.layers.Dense(256, activation='selu'))
    # value.add(K.layers.Dense(1, activation='linear'))
    vf = K.layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    vf = K.layers.Dense(64, activation=tf.nn.leaky_relu)(vf)
    vf = K.layers.Dense(1, activation="linear")(vf)
    value = K.models.Model(inputs=input, outputs=vf)

    try:
        agent.load_weights(f"models/{BOARD_SIZE}x{BOARD_SIZE}_bigger/agent")
        value.load_weights(f"models/{BOARD_SIZE}x{BOARD_SIZE}_bigger/value")
        ae.load_weights(f"models/{BOARD_SIZE}x{BOARD_SIZE}_bigger/ae")
    except:
        pass
    # endregion
    env_ = NumpyEnvironment(1000, BOARD_SIZE)#, chunk_size=10, cpu_threads_multiplier=1)
    GAMMA = .99

    optimizer_value = K.optimizers.Adam(1e-4)
    optimizer_agent = K.optimizers.Adam(1e-4)
    optimizer_autoencoder = K.optimizers.Adam(1e-5)

    avg_rewards = []
    ae_losses = []

    for iterations in trange(ITERATIONS):
        state = env_.to_state()

        if iterations % 10 == 0:
            agent.save_weights(f"models/{BOARD_SIZE}x{BOARD_SIZE}_bigger/agent")
            value.save_weights(f"models/{BOARD_SIZE}x{BOARD_SIZE}_bigger/value")
            ae.save_weights(f"models/{BOARD_SIZE}x{BOARD_SIZE}_bigger/ae")
        with tf.GradientTape(persistent=True) as tape:
            # calculate distributions of actions
            original_probs = agent(
                state
            )
            # remove actions that are not available
            probs = re_normalize_possible_actions(
                state,
                original_probs
            )
            # sample actions
            actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)

            # MDP update
            # print("start move")
            rewards = env_.move(actions)
            # print("end move")
            new_state = env_.to_state()

            # TD error
            value_state = value(state)
            td_error = tf.stop_gradient((rewards + GAMMA * value(new_state, training=False)) - value_state) * -1  # to do gradient ascend

            # calculate the loss for both value and agent
            actions_indexes = tf.concat((tf.range(actions.shape[0])[..., None], actions), axis=-1)

            # maybe introduce eligibility trace to simulate n-step td, to have longer dependencies
            loss_agent = tf.stop_gradient(td_error) * tf.math.log(tf.gather_nd(probs, actions_indexes))
            loss_value = tf.stop_gradient(td_error) * value_state
            rec = ae(state)
            loss = tf.reduce_sum(K.losses.CategoricalCrossentropy()(state, rec))
        ae_losses.append(loss)
        grad = tape.gradient(loss, ae.trainable_weights)
        optimizer_autoencoder.apply_gradients(zip(grad, ae.trainable_weights))

        # calculate gradient
        gradient_agent = tape.gradient(loss_agent, agent.trainable_weights)
        gradient_value = tape.gradient(loss_value, value.trainable_weights)
        avg_rewards.append(tf.reduce_mean(rewards))

        # update neural nets weights
        optimizer_agent.apply_gradients(zip(gradient_agent, agent.trainable_weights))
        optimizer_value.apply_gradients(zip(gradient_value, value.trainable_weights))

        # just to be sure that long trainings do not end for some strange reason...
        # except Exception as e:
        #     print(e)
        #     print("Resetting env")
        #     env_ = ProcessEnvironment(1, BOARD_SIZE)



    random_env_ = NumpyEnvironment(100, BOARD_SIZE)
    random_rewards = []

    for _ in trange(100):
        state = random_env_.to_state()
        probs = re_normalize_possible_actions(
            state,
            tf.repeat([[.25] * 4], 100, axis=0)
        )
        # sample actions
        actions = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

        # MDP update
        rewards = random_env_.move(actions)
        random_rewards.append(tf.reduce_mean(rewards))

    plt.plot(ae_losses)

    plt.plot(np.array(avg_rewards).reshape((-1, 10)).mean(axis=-1))
    n = np.array(avg_rewards).reshape((-1, 10)).mean(axis=-1).shape[0]
    plt.plot(np.arange(0, n), np.repeat(np.mean(random_rewards), n))
    _ = plt.xlabel("iterations")
    _ = plt.ylabel("avg reward")
