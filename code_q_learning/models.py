import tensorflow as tf
import keras.api._v2.keras as K


def load_models(env_, folder_name=None):
    input = K.layers.Input(shape=(env_.board_size, env_.board_size, 4))
    x = K.layers.Conv2D(16, (3, 3), activation="linear", padding="SAME")(input)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.MaxPool2D(2, padding="same")(x)

    if env_.board_size > 8:
        x = K.layers.Conv2D(16, (3, 3), activation="linear", padding="SAME")(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
    if env_.board_size > 12:
        x = K.layers.MaxPool2D(2, padding="same")(x)
        x = K.layers.Conv2D(16, (3, 3), activation="linear", padding="SAME")(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
    if env_.board_size > 16:
        x = K.layers.Conv2D(16, (3, 3), activation="linear", padding="SAME")(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
    if env_.board_size > 20:
        x = K.layers.MaxPool2D(2, padding="same")(x)
        x = K.layers.Conv2D(16, (3, 3), activation="linear", padding="SAME")(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Flatten()(x)
    q = K.layers.Dense(256, activation=tf.nn.leaky_relu)(x)
    q = K.layers.Dense(256, activation=tf.nn.leaky_relu)(q)
    q = K.layers.Dense(4, activation="linear")(q)
    qfunction = K.models.Model(inputs=input, outputs=q)

    avg_rewards = []

    if folder_name is not None:
        try:
            qfunction.load_weights(folder_name + f"/qfunction")
            print("loaded")
        except:
            pass
        try :
            import json
            with open(f"{folder_name}/training.txt", "r") as file:
                avg_rewards = json.load(file)
            print("loaded")
        except:
            pass

    return qfunction, avg_rewards
