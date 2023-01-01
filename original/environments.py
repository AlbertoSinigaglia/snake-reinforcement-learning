import sys, os
import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

import multiprocessing.dummy as dummy
from multiprocessing import cpu_count
from itertools import islice
from itertools import repeat

class BaseEnvironment:
    def __init__(self, n_boards, board_size):
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        self.WIN_REWARD = 20.
        self.FRUIT_REWARD = 10.
        self.STEP_REWARD = 0.
        self.ATE_HIMSELF_REWARD = 10.  # scalar to multiply to -len(eaten body)

        self.HEAD = 3
        self.BODY = 2
        self.FRUIT = 1
        self.EMPTY = 0

        self.board_size = board_size
        self.n_boards = n_boards
        self.boards = np.zeros((self.n_boards, self.board_size, self.board_size))
        for board in self.boards:
            i = np.random.randint(0, self.board_size)
            j = np.random.randint(0, self.board_size)
            board[i, j] = self.HEAD
            while board[i, j] != 0:
                i = np.random.randint(0, self.board_size)
                j = np.random.randint(0, self.board_size)
            board[i, j] = self.FRUIT
        self.bodies = [[] for _ in range(self.n_boards)]

    def to_state(self):
        return K.utils.to_categorical(self.boards)[..., 1:]

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
                rewards[b] = -len(to_delete) * self.ATE_HIMSELF_REWARD

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
