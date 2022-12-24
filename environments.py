import sys, os
import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

import multiprocessing.dummy as dummy
from multiprocessing import cpu_count
from itertools import islice
from itertools import repeat


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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


class BaseEnvironment:
    def __init__(self, n_boards, board_size):
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        self.WIN_REWARD = 10000
        self.FRUIT_REWARD = 10
        self.STEP_REWARD = 0
        self.ATE_HIMSELF_REWARD = 10  # scalar to multiply to -len(eaten body)

        self.HEAD = 3
        self.BODY = 2
        self.FRUIT = 1
        self.EMPTY = 0

        self.board_size = board_size
        self.n_boards = n_boards
        self.boards = np.zeros((self.n_boards, self.board_size, self.board_size))
        for board in self.boards:
            i = np.random.randint(0, self.board_size);
            j = np.random.randint(0, self.board_size)
            board[i, j] = self.HEAD
            while board[i, j] != 0:
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
            board[i, j] = self.FRUIT
        self.bodies = [[] for _ in range(self.n_boards)]

    def to_state(self):
        return K.utils.to_categorical(self.boards)[..., 1:]


class SequentialEnvironment(BaseEnvironment):
    def __init__(self, n_boards, board_size):
        super().__init__(n_boards, board_size)

    def move(self, actions):
        rewards = []
        # [up, right, bottom, left]
        for board_count, (board, action, body) in enumerate(zip(self.boards, actions, self.bodies)):
            if tf.reduce_sum(board) - 5 == (self.board_size ** 2 - 3) * self.BODY:
                print("won")
                rewards.append(self.WIN_REWARD)
                board = np.zeros(self.board_size, self.board_size)
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
                board[i, j] = self.HEAD
                while board[i, j] != 0:
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                board[i, j] = self.FRUIT
                body = []

            dx = 0;
            dy = 0
            if action in [self.UP, self.DOWN]:
                dx = 1 if action == self.UP else -1
            if action in [self.RIGHT, self.LEFT]:
                dy = 1 if action == self.RIGHT else -1
            head = tf.cast(tf.where(board == self.HEAD)[0], dtype=tf.int32)
            fruit = tf.cast(tf.where(board == self.FRUIT)[0], dtype=tf.int32)

            body.insert(0, head)
            board[tuple(head.numpy())] = self.BODY

            # eat fruit
            if tf.reduce_all(tf.equal(head + [dx, dy], fruit)):
                board[tuple(fruit.numpy())] = self.EMPTY
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
                while board[i, j] != 0 and tf.reduce_all(tf.equal(head + [dx, dy], [i, j])):
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                board[i, j] = self.FRUIT
                rewards.append(self.FRUIT_REWARD)
            # he eats himself
            elif tf.reduce_any(
                    tf.reduce_all(tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                                  axis=-1)):
                since = tf.where(
                    tf.reduce_all(tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                                  axis=-1))[0, 0]
                pieces_eaten = len(body) - since
                for pos in body[since:]:
                    board[tuple(pos)] = self.EMPTY
                del body[since:]
                position = body.pop()
                board[tuple(position.numpy())] = self.EMPTY
                rewards.append(-pieces_eaten * self.ATE_HIMSELF_REWARD)
            # doesn't eat fruit
            else:
                position = body.pop()
                board[tuple(position.numpy())] = self.EMPTY
                rewards.append(self.STEP_REWARD)

            board[tuple(head + [dx, dy])] = self.HEAD

            if tf.size(tf.where(board == self.FRUIT)) == 0:
                # probably better to check the indexes where the board is free and pick one at random
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
                while board[i, j] != 0:
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                board[i, j] = self.FRUIT

        return tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))


class ProcessEnvironment(BaseEnvironment):
    def __init__(self, n_boards, board_size):
        super().__init__(n_boards, board_size)

    def move(self, actions):
        rewards = [0] * self.n_boards

        def process_board(elems):
            board_count, (board, action, body) = elems
            if tf.reduce_sum(board) - 5 == (self.board_size ** 2 - 3) * self.BODY:
                print("won")
                rewards[board_count] = self.WIN_REWARD
                board = np.zeros(self.board_size, self.board_size)
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
                board[i, j] = self.HEAD
                while board[i, j] != 0:
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                board[i, j] = self.FRUIT
                del body[:]

            dx = 0;
            dy = 0
            if action in [self.UP, self.DOWN]:
                dx = 1 if action == self.UP else -1
            if action in [self.RIGHT, self.LEFT]:
                dy = 1 if action == self.RIGHT else -1
            head = tf.cast(tf.where(board == self.HEAD)[0], dtype=tf.int32)
            fruit = tf.cast(tf.where(board == self.FRUIT)[0], dtype=tf.int32)

            body.insert(0, head)
            board[tuple(head.numpy())] = self.BODY

            # eat fruit
            if tf.reduce_all(tf.equal(head + [dx, dy], fruit)):
                board[tuple(fruit.numpy())] = self.EMPTY
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
                while board[i, j] != 0 and tf.reduce_all(tf.equal(head + [dx, dy], [i, j])):
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                board[i, j] = self.FRUIT
                rewards[board_count] = self.FRUIT_REWARD
            # he eats himself
            elif tf.reduce_any(
                    tf.reduce_all(tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                                  axis=-1)):
                since = tf.where(
                    tf.reduce_all(tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                                  axis=-1))[0, 0]
                pieces_eaten = len(body) - since
                for pos in body[since:]:
                    board[tuple(pos)] = self.EMPTY
                del body[since:]
                position = body.pop()
                board[tuple(position.numpy())] = self.EMPTY
                rewards[board_count] = -pieces_eaten * self.ATE_HIMSELF_REWARD
            # doesn't eat fruit
            else:
                position = body.pop()
                board[tuple(position.numpy())] = self.EMPTY
                rewards[board_count] = self.STEP_REWARD

            board[tuple(head + [dx, dy])] = self.HEAD

            if tf.size(tf.where(board == self.FRUIT)) == 0:
                # probably better to check the indexes where the board is free and pick one at random
                i = np.random.randint(0, self.board_size);
                j = np.random.randint(0, self.board_size)
                while board[i, j] != 0:
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                board[i, j] = self.FRUIT

        with dummy.Pool(cpu_count()) as p:
            _ = p.map(process_board, enumerate(zip(self.boards, actions, self.bodies)))
        return tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))


class ChunkProcessEnvironment(BaseEnvironment):
    def __init__(self, n_boards, board_size, chunk_size=25):
        super().__init__(n_boards, board_size)
        self.chunk_size = chunk_size
        self.pool = dummy.Pool(cpu_count())

    def close_pool(self):
        self.pool.close()

    def reset_pool(self):
        self.close_pool()
        self.pool = Pool(cpu_count())

    def __del__(self):
        self.close_pool()

    def move(self, actions):
        rewards = [0] * self.n_boards

        def process_board(elems):
            for board_count, board, action, body in elems:
                if tf.reduce_sum(board) - 5 == (self.board_size ** 2 - 3) * self.BODY:
                    print("won")
                    rewards[board_count] = self.WIN_REWARD
                    board = np.zeros(self.board_size, self.board_size)
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                    board[i, j] = self.HEAD
                    while board[i, j] != 0:
                        i = np.random.randint(0, self.board_size);
                        j = np.random.randint(0, self.board_size)
                    board[i, j] = self.FRUIT
                    del body[:]

                dx = 0;
                dy = 0
                if action in [self.UP, self.DOWN]:
                    dx = 1 if action == self.UP else -1
                if action in [self.RIGHT, self.LEFT]:
                    dy = 1 if action == self.RIGHT else -1
                head = tf.cast(tf.where(board == self.HEAD)[0], dtype=tf.int32)
                fruit = tf.cast(tf.where(board == self.FRUIT)[0], dtype=tf.int32)

                body.insert(0, head)
                board[tuple(head.numpy())] = self.BODY

                # eat fruit
                if tf.reduce_all(tf.equal(head + [dx, dy], fruit)):
                    board[tuple(fruit.numpy())] = self.EMPTY
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                    while board[i, j] != 0 and tf.reduce_all(tf.equal(head + [dx, dy], [i, j])):
                        i = np.random.randint(0, self.board_size);
                        j = np.random.randint(0, self.board_size)
                    board[i, j] = self.FRUIT
                    rewards[board_count] = self.FRUIT_REWARD
                # he eats himself
                elif tf.reduce_any(tf.reduce_all(
                        tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                        axis=-1)):
                    since = tf.where(tf.reduce_all(
                        tf.equal(body, tf.repeat(tf.expand_dims(head + [dx, dy], axis=0), len(body), axis=0)),
                        axis=-1))[0, 0]
                    pieces_eaten = len(body) - since
                    for pos in body[since:]:
                        board[tuple(pos)] = self.EMPTY
                    del body[since:]
                    position = body.pop()
                    board[tuple(position.numpy())] = self.EMPTY
                    rewards[board_count] = -pieces_eaten * self.ATE_HIMSELF_REWARD
                # doesn't eat fruit
                else:
                    position = body.pop()
                    board[tuple(position.numpy())] = self.EMPTY
                    rewards[board_count] = self.STEP_REWARD

                board[tuple(head + [dx, dy])] = self.HEAD

                if tf.size(tf.where(board == self.FRUIT)) == 0:
                    # probably better to check the indexes where the board is free and pick one at random
                    i = np.random.randint(0, self.board_size);
                    j = np.random.randint(0, self.board_size)
                    while board[i, j] != 0:
                        i = np.random.randint(0, self.board_size);
                        j = np.random.randint(0, self.board_size)
                    board[i, j] = self.FRUIT

        _ = self.pool.map(process_board,
                          chunk(zip(np.arange(0, self.n_boards), self.boards, actions, self.bodies), self.chunk_size))
        return tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))


class ThreadEnvironment(BaseEnvironment):
    def __init__(self, n_boards, board_size, chunk_size=25, cpu_threads_multiplier=1):
        super().__init__(n_boards, board_size)
        self.chunk_size = chunk_size
        self.cpu_threads_multiplier = cpu_threads_multiplier
        self.pool = Pool(cpu_count() * self.cpu_threads_multiplier)

    def close_pool(self):
        self.pool.close()

    def reset_pool(self):
        self.close_pool()
        self.pool = Pool(cpu_count() * self.cpu_threads_multiplier)

    def __del__(self):
        self.close_pool()

    def move(self, actions):
        rewards = self.pool.map(process_boards,
                                chunk(zip(
                                    repeat({
                                        "UP": self.UP,
                                        "DOWN": self.DOWN,
                                        "LEFT": self.LEFT,
                                        "RIGHT": self.RIGHT,
                                        "HEAD": self.HEAD,
                                        "BODY": self.BODY,
                                        "FRUIT": self.FRUIT,
                                        "EMPTY": self.EMPTY,
                                        "WIN_REWARD": self.WIN_REWARD,
                                        "FRUIT_REWARD": self.FRUIT_REWARD,
                                        "STEP_REWARD": self.STEP_REWARD,
                                        "ATE_HIMSELF_REWARD": self.ATE_HIMSELF_REWARD,
                                        "board_size": self.board_size

                                    }, times=len(self.boards)),
                                    self.boards,
                                    actions,
                                    self.bodies
                                ), self.chunk_size))

        return tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))


class NumpyEnvironment(BaseEnvironment):
    def __init__(self, n_boards, board_size):
        super().__init__(n_boards, board_size)

    def move(self, actions):
        rewards = np.zeros(self.n_boards, dtype=float)
        dx = np.zeros(len(actions))
        dx[np.where(actions == self.UP)[0]] = 1
        dx[np.where(actions == self.DOWN)[0]] = -1
        dy = np.zeros(len(actions))
        dy[np.where(actions == self.RIGHT)[0]] = 1
        dy[np.where(actions == self.LEFT)[0]] = -1
        # new heads per board
        heads = np.argwhere(self.boards == self.HEAD)
        offset = np.hstack((np.zeros_like(actions), dx[:, None], dy[:, None]))
        new_heads = (heads + offset).astype(int)

        # fruits per board
        fruits = np.argwhere(self.boards == self.FRUIT)
        fruits_eaten_bool = np.all(fruits == new_heads, axis=-1)
        boards_where_fruits_is_been_eaten = np.argwhere(fruits_eaten_bool)

        # ate himself
        bodies_ = np.argwhere(self.boards == self.BODY)
        bodies_head = new_heads[bodies_[:, 0]]
        boards_where_bodies_is_been_eaten = bodies_head[:, 0]
        if len(bodies_head):
            bodies_eaten_bool = np.all(bodies_ == bodies_head, axis=-1)
            eaten_body_peace = bodies_[np.argwhere(bodies_eaten_bool)[:, 0]]
            for piece in eaten_body_peace:
                b = piece[0]
                piece = piece[1:]
                index = np.argwhere(np.all(np.array(self.bodies[b]) == piece, axis=-1))[0, 0]
                to_delete = self.bodies[b][index:]
                to_delete_np = np.array(to_delete)
                self.boards[b, to_delete_np[:, 0], to_delete_np[:, 1]] = self.EMPTY
                del self.bodies[b][index:]

                rewards[b] = -(1 + index) * self.ATE_HIMSELF_REWARD

        # remove last peace of each body (if fruit not been eaten) and add the head
        for i in range(self.n_boards):
            self.bodies[i].insert(0, heads[i][1:])
            self.boards[heads[i][0], heads[i][1], heads[i][2]] = self.BODY
            if i not in boards_where_fruits_is_been_eaten.reshape((-1,)).tolist():
                self.boards[heads[i][0], self.bodies[i][-1][0], self.bodies[i][-1][1]] = self.EMPTY
                self.bodies[i].pop()

        boards_where_fruits_is_been_eaten = boards_where_fruits_is_been_eaten.reshape((-1,))
        self.boards[
            heads[boards_where_fruits_is_been_eaten][:, 0],
            heads[boards_where_fruits_is_been_eaten][:, 1],
            heads[boards_where_fruits_is_been_eaten][:, 2]] = self.BODY
        self.boards[new_heads[:, 0], new_heads[:, 1], new_heads[:, 2]] = self.HEAD

        rewards[boards_where_fruits_is_been_eaten] += self.FRUIT_REWARD
        rewards[np.setdiff1d(
            np.arange(0, self.n_boards),
            np.union1d(boards_where_fruits_is_been_eaten, boards_where_bodies_is_been_eaten))
        ] += self.STEP_REWARD

        # check add fruit to boards where it's been eaten
        for b in boards_where_fruits_is_been_eaten:
            available = np.argwhere(self.boards[b] == self.EMPTY)
            if len(available) == 0:
                print("won")
                self.boards[b] = np.zeros((self.board_size, self.board_size))
                self.bodies[b] = []
                rewards[b] = self.WIN_REWARD
                i = np.random.randint(0, self.board_size)
                j = np.random.randint(0, self.board_size)
                self.boards[b][i, j] = self.HEAD

            available = np.argwhere(self.boards[b] == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            self.boards[b][ind[0], ind[1]] = self.FRUIT

        return tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))
