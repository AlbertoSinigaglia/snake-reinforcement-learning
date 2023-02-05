import tensorflow as tf
import keras.api._v2.keras as K


def load_models(env_, folder_name=None):
    input = K.layers.Input(shape=(env_.board_size, env_.board_size, 4))
    x = K.layers.Conv2D(64, (3, 3), activation="linear", padding="SAME")(input)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    if env_.board_size > 16:
        x = K.layers.MaxPool2D(2, padding="same")(x)
        x = K.layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu, padding="SAME")(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
    if env_.board_size > 20:
        x = K.layers.MaxPool2D(2, padding="same")(x)
        x = K.layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu, padding="SAME")(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Flatten()(x)
    x = K.layers.Dense(256, activation="linear")(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    vf = K.layers.Dense(256, activation="linear")(x)
    vf = K.layers.Activation(tf.nn.leaky_relu)(vf)
    vf = K.layers.Dense(1, activation="linear")(vf)

    af = K.layers.Dense(256, activation="linear")(x)
    af = K.layers.Activation(tf.nn.leaky_relu)(af)
    af = K.layers.Dense(4, activation="linear")(af)

    qfunction = K.models.Model(inputs=input, outputs=[vf, af])

    avg_rewards = []

    if folder_name is not None:
        try:
            qfunction.load_weights(folder_name + f"/agent")
            print("loaded")
        except:
            pass
        try:
            import json
            with open(f"{folder_name}/training.txt", "r") as file:
                avg_rewards = json.load(file)
            print("loaded")
        except:
            pass

    return qfunction, avg_rewards
