import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np


class BaseEnvironment:
    HEAD = 4
    BODY = 3
    FRUIT = 2
    EMPTY = 1
    WALL = 0

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4

    def __init__(self, n_boards, board_size):
        self.WIN_REWARD = 1.
        self.FRUIT_REWARD = .5
        self.STEP_REWARD = 0.
        self.ATE_HIMSELF_REWARD = -.2
        self.HIT_WALL_REWARD = -.1

        self.board_size = board_size
        self.n_boards = n_boards

        # create the actual boards (empty, subclasses will fill them with the required stuff)
        self.boards = np.ones((self.n_boards, self.board_size, self.board_size)) * self.EMPTY
        # store the bodies of the snakes
        self.bodies = [[] for _ in range(self.n_boards)]

    """
    Given the set of new heads, checks if those would fall over a wall
    returns: only the heads that bumped into a wall
    """
    def check_actions(self, new_heads):
        walls = np.argwhere(self.boards == BaseEnvironment.WALL)
        # using a sort of hash function (counting position in base "np.max(walls)+1" instead of base 10)
        powers = (np.max(walls) + 1) ** np.arange(walls.shape[1])
        walls_code = np.sum(walls * powers, axis=-1)
        new_heads_code = np.sum(new_heads * powers, axis=-1)
        return new_heads[np.isin(new_heads_code, walls_code)][:, 0]

    def move(self, actions):
        # find heads in the boards
        heads = np.argwhere(self.boards == self.HEAD)
        actions = np.array(actions)
        # init rewards
        rewards = np.zeros(self.n_boards, dtype=float)
        # calculate action offset (from 0,1,2,3 to +1/-1 in x/y)
        dx = np.zeros(len(actions))
        dx[np.where(actions == self.UP)[0]] = 1
        dx[np.where(actions == self.DOWN)[0]] = -1
        dy = np.zeros(len(actions))
        dy[np.where(actions == self.RIGHT)[0]] = 1
        dy[np.where(actions == self.LEFT)[0]] = -1
        offset = np.hstack((np.zeros_like(actions), dx[:, None], dy[:, None]))
        # new heads per board
        new_heads = (heads + offset).astype(int)
        # find heads that hit the wall, and for those set the new_head to the current one (no move)
        # and the reward
        hit_wall = self.check_actions(new_heads)
        actions[hit_wall] = self.NONE
        new_heads[hit_wall] = heads[hit_wall]
        dx[hit_wall] = 0.
        dy[hit_wall] = 0.
        offset[hit_wall] = 0.
        rewards[hit_wall] = self.HIT_WALL_REWARD

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
            # find the peaces that are in the same place as the new head (ate himself)
            eaten_body_peace = bodies_[np.argwhere(bodies_eaten_bool)[:, 0]]
            # for each one of them, delete the trailing tail
            for piece in eaten_body_peace:
                b = piece[0]
                piece = piece[1:]
                index = np.argwhere(np.all(np.array(self.bodies[b]) == piece, axis=-1))[0, 0]
                to_delete = self.bodies[b][index:]
                to_delete_np = np.array(to_delete)
                self.boards[b, to_delete_np[:, 0], to_delete_np[:, 1]] = self.EMPTY
                del self.bodies[b][index:]
                rewards[b] = self.ATE_HIMSELF_REWARD  # * len(to_delete)

        # remove last peace of each body (if fruit not been eaten) and add the head
        for i in range(self.n_boards):
            self.bodies[i].insert(0, heads[i][1:])
            self.boards[i][np.where(self.boards[i] == self.BODY)] = self.EMPTY
            self.boards[i][heads[i, 1], heads[i, 2]] = self.EMPTY
            if i not in boards_where_fruits_is_been_eaten.reshape((-1,)).tolist():
                self.bodies[i].pop()
            if self.bodies[i]:
                body = np.array(self.bodies[i])
                self.boards[i][body[:, 0], body[:, 1]] = self.BODY

        boards_where_fruits_is_been_eaten = boards_where_fruits_is_been_eaten.reshape((-1,))
        # set body where fruit has been eaten
        self.boards[
            heads[boards_where_fruits_is_been_eaten][:, 0],
            heads[boards_where_fruits_is_been_eaten][:, 1],
            heads[boards_where_fruits_is_been_eaten][:, 2]] = self.BODY
        # set new heads position
        self.boards[new_heads[:, 0], new_heads[:, 1], new_heads[:, 2]] = self.HEAD

        rewards[boards_where_fruits_is_been_eaten] = self.FRUIT_REWARD
        rewards[np.setdiff1d(
            np.arange(0, self.n_boards),
            np.union1d(boards_where_fruits_is_been_eaten, boards_where_bodies_is_been_eaten))
        ] = self.STEP_REWARD

        # (check) add fruit to boards where it's been eaten
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

            chosen = available[np.random.choice(range(len(available)))]
            self.boards[b][chosen[0], chosen[1]] = self.FRUIT

        return tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))

    def to_state(self):
        return K.utils.to_categorical(self.boards)[..., 1:]


class Walls25x25SnakeEnvironment(BaseEnvironment):
    BOARD_SIZE = 25

    def __init__(self, n_boards):
        super().__init__(n_boards, self.BOARD_SIZE)
        # vertical and horizontal walls
        self.boards[:, [0, 8, 16, 24], :] = self.WALL
        self.boards[:, :, [0, 8, 16, 24]] = self.WALL

        # available walls to be removed
        available_walls = np.array([
            # vertical walls between rooms
            [[8, 2], [8, 3], [8, 4], [8, 5], [8, 6]],  # [8,1],[8,7]
            [[8, 10], [8, 11], [8, 12], [8, 13], [8, 14]],  # [8,9],[8,15]
            [[8, 18], [8, 19], [8, 20], [8, 21], [8, 22]],  # [8,17],[8,23]
            [[16, 2], [16, 3], [16, 4], [16, 5], [16, 6]],  # [16,1],[16,7]
            [[16, 10], [16, 11], [16, 12], [16, 13], [16, 14]],  # [16,9],[16,15]
            [[16, 18], [16, 19], [16, 20], [16, 21], [16, 22]],  # [16,17],[8,23]
            # horizontal walls between rooms
            [[2, 8], [3, 8], [4, 8], [5, 8], [6, 8]],  # [1,8],[7,8]
            [[10, 8], [11, 8], [12, 8], [13, 8], [14, 8]],  # [9,8],[15,8]
            [[18, 8], [19, 8], [20, 8], [21, 8], [22, 8]],  # [17,8],[23,8]
            [[2, 16], [3, 16], [4, 16], [5, 16], [6, 16]],  # [1,16],[7,16]
            [[10, 16], [11, 16], [12, 16], [13, 16], [14, 16]],  # [9,16],[15,16]
            [[18, 16], [19, 16], [20, 16], [21, 16], [22, 16]],  # [17,16],[23,8]
        ])
        for board in self.boards:
            # pick a wall for each segment, remove it, add head, add fruit
            indexes = available_walls[
                np.arange(available_walls.shape[0]), np.random.randint(0, available_walls.shape[1], 12)]
            board[indexes[:, 0], indexes[:, 1]] = self.EMPTY

            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.HEAD
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.FRUIT


class Walls17x17SnakeEnvironment(BaseEnvironment):
    BOARD_SIZE = 17

    def __init__(self, n_boards):
        super().__init__(n_boards, self.BOARD_SIZE)
        self.board_size = self.BOARD_SIZE
        self.n_boards = n_boards
        self.boards = np.ones((self.n_boards, self.board_size, self.board_size)) * self.EMPTY
        # vertical and horizontal walls
        self.boards[:, [0, 8, 16], :] = self.WALL
        self.boards[:, :, [0, 8, 16]] = self.WALL
        # available walls to be removed
        available_walls = np.array([
            # vertical walls between rooms
            [[8, 2], [8, 3], [8, 4], [8, 5], [8, 6]],  # [8,1],[8,7]
            [[8, 10], [8, 11], [8, 12], [8, 13], [8, 14]],  # [8,9],[8,15]
            # horizontal walls between rooms
            [[2, 8], [3, 8], [4, 8], [5, 8], [6, 8]],  # [1,8],[7,8]
            [[10, 8], [11, 8], [12, 8], [13, 8], [14, 8]],  # [9,8],[15,8]
        ])
        for board in self.boards:
            indexes = available_walls[
                np.arange(available_walls.shape[0]), np.random.randint(0, available_walls.shape[1], 4)]
            board[indexes[:, 0], indexes[:, 1]] = self.EMPTY

            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.HEAD
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.FRUIT


class Walls9x9SnakeEnvironment(BaseEnvironment):
    BOARD_SIZE = 9

    def __init__(self, n_boards):
        super().__init__(n_boards, self.BOARD_SIZE)
        self.board_size = self.BOARD_SIZE
        self.n_boards = n_boards
        self.boards = np.ones((self.n_boards, self.board_size, self.board_size)) * self.EMPTY
        # vertical and horizontal walls
        self.boards[:, [0, 4, 8], :] = self.WALL
        self.boards[:, :, [0, 4, 8]] = self.WALL
        # available walls to be removed
        available_walls = np.array([
            # vertical walls between rooms
            [[4, 1], [4, 2], [4, 3]],
            [[4, 5], [4, 6], [4, 7]],
            # horizontal walls between rooms
            [[1, 4], [2, 4], [3, 4]],
            [[5, 4], [6, 4], [7, 4]],
        ])
        for board in self.boards:
            # pick a wall for each segment, remove it, add head, add fruit
            indexes = available_walls[
                np.arange(available_walls.shape[0]), np.random.randint(0, available_walls.shape[1], 4)]
            board[indexes[:, 0], indexes[:, 1]] = self.EMPTY

            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.HEAD
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.FRUIT


class OriginalSnakeEnvironment(BaseEnvironment):
    def __init__(self, n_boards, board_size):
        super().__init__(n_boards, board_size)
        # boards
        self.boards[:, [0, -1], :] = self.WALL
        self.boards[:, :, [0, -1]] = self.WALL
        for board in self.boards:
            # add head, add fruit
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.HEAD
            available = np.argwhere(board == self.EMPTY)
            ind = available[np.random.choice(range(len(available)))]
            board[ind[0], ind[1]] = self.FRUIT
