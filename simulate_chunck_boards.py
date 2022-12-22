import sys, os
import tensorflow as tf
import numpy as np



def process_boards(boards):
    rewards = []
    for elems in boards:
        env, board, action, body = elems
        if tf.reduce_sum(board) - 5 == (env["board_size"] ** 2 - 3) * env["BODY"]:
            print("won")
            rewards.append(env["WIN_REWARD"])
            board = np.zeros(env["board_size"], env["board_size"])
            i = np.random.randint(0, env["board_size"])
            j = np.random.randint(0, env["board_size"])
            board[i, j] = env["HEAD"]
            while board[i, j] != 0:
                i = np.random.randint(0, env["board_size"])
                j = np.random.randint(0, env["board_size"])
            board[i, j] = env["FRUIT"]
            del body[:]

        dx = 0
        dy = 0
        if action in [env["UP"], env["DOWN"]]:
            dx = 1 if action == env["UP"] else -1
        if action in [env["RIGHT"], env["LEFT"]]:
            dy = 1 if action == env["RIGHT"] else -1
        head = tf.cast(tf.where(board == env["HEAD"])[0], dtype=tf.int32)
        fruit = tf.cast(tf.where(board == env["FRUIT"])[0], dtype=tf.int32)

        body.insert(0, head)
        board[tuple(head.numpy())] = env["BODY"]

        # eat fruit
        if tf.reduce_all(tf.equal(head + [dx, dy], fruit)):
            board[tuple(fruit.numpy())] = env["EMPTY"]
            i = np.random.randint(0, env["board_size"])
            j = np.random.randint(0, env["board_size"])
            while board[i, j] != 0 and tf.reduce_all(tf.equal(head + [dx, dy], [i, j])):
                i = np.random.randint(0, env["board_size"])
                j = np.random.randint(0, env["board_size"])
            board[i, j] = env["FRUIT"]
            rewards.append(env["FRUIT_REWARD"])
        # he eats himself
        elif tf.reduce_any(
                tf.reduce_all(tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                              axis=-1)):
            since = tf.where(
                tf.reduce_all(tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                              axis=-1))[0, 0]
            pieces_eaten = len(body) - since
            for pos in body[since:]:
                board[tuple(pos)] = env["EMPTY"]
            del body[since:]
            position = body.pop()
            board[tuple(position.numpy())] = env["EMPTY"]
            rewards.append(-pieces_eaten * env["ATE_HIMSELF_REWARD"])
        # doesn't eat fruit
        else:
            position = body.pop()
            board[tuple(position.numpy())] = env["EMPTY"]
            rewards.append(env["STEP_REWARD"])

        board[tuple(head + [dx, dy])] = env["HEAD"]

        if tf.size(tf.where(board == env["FRUIT"])) == 0:
            # probably better to check the indexes where the board is free and pick one at random
            i = np.random.randint(0, env["board_size"])
            j = np.random.randint(0, env["board_size"])
            while board[i, j] != 0:
                i = np.random.randint(0, env["board_size"])
                j = np.random.randint(0, env["board_size"])
            board[i, j] = env["FRUIT"]
    return rewards
#%%
