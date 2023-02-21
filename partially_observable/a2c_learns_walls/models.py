import tensorflow as tf
import keras.api._v2.keras as K


def load_models(env_, folder_name=None):
    input = K.layers.Input(shape=(env_.mask_size * 2 + 1, env_.mask_size * 2 + 1, 4))
    policy = K.layers.Flatten()(input)
    policy = K.layers.Dense(256, activation="linear")(policy)
    policy = K.layers.Activation(tf.nn.leaky_relu)(policy)
    policy = K.layers.Dense(256, activation="linear")(policy)
    policy = K.layers.Activation(tf.nn.leaky_relu)(policy)
    policy = K.layers.Dense(4, activation=tf.nn.softmax)(policy)
    agent = K.models.Model(inputs=input, outputs=policy)

    input = K.layers.Input(shape=(env_.mask_size * 2 + 1, env_.mask_size * 2 + 1, 4))
    vf = K.layers.Flatten()(input)
    vf = K.layers.Dense(256, activation="linear")(vf)
    vf = K.layers.Activation(tf.nn.leaky_relu)(vf)
    vf = K.layers.Dense(256, activation="linear")(vf)
    vf = K.layers.Activation(tf.nn.leaky_relu)(vf)
    vf = K.layers.Dense(1, activation="linear")(vf)
    value = K.models.Model(inputs=input, outputs=vf)

    avg_rewards = []

    if folder_name is not None:
        try:
            agent.load_weights(folder_name + f"/agent")
            value.load_weights(folder_name + f"/value")
            print("loaded 1")
        except Exception as e:
            print(e)
            pass
        try:
            import json
            with open(f"{folder_name}/training.txt", "r") as file:
                avg_rewards = json.load(file)
            print("loaded 2")
        except:
            pass

    return agent, value, avg_rewards
