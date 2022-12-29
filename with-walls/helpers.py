import multiprocessing

from environments import *
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count


def reinitialize(model):
    for l in model.layers:
        if hasattr(l, "kernel_initializer"):
            try:
                l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
            except:
                pass
        if hasattr(l, "bias_initializer"):
            try:
                l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
            except:
                pass
        if hasattr(l, "recurrent_initializer"):
            try:
                l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            except:
                pass


pool = Pool(cpu_count())


def process_heads(params):
    heads, walls = params
    indexes = []
    for head in heads:
        t = np.argwhere(np.all(walls == head, axis=-1, keepdims=False)).reshape((-1,))
        if len(t):
            indexes.append(t)
    return indexes


def re_normalize_possible_actions(state, probs, mask_with=0.):
    global pool
    boards = tf.argmax(state, axis=-1)
    heads = np.argwhere(boards == BaseEnvironment.HEAD)
    walls = np.argwhere(boards == BaseEnvironment.WALL)
    mask = np.ones_like(probs)

    heads_up = np.copy(heads)
    heads_down = np.copy(heads)
    heads_left = np.copy(heads)
    heads_right = np.copy(heads)
    heads_up[:, 1] += 1
    heads_down[:, 1] -= 1
    heads_left[:, 2] += 1
    heads_right[:, 2] -= 1

    # indexes_left = []
    # indexes_right = []
    # indexes_down = []
    # indexes_up = []
    #
    # for head_down, head_up, head_left,head_right in zip(heads_down, heads_up, heads_left,heads_right):
    #     t = np.argwhere(np.all(walls == head_left, axis=-1, keepdims=False)).reshape((-1,))
    #     if len(t):
    #         indexes_left.append(t)
    #
    #     t = np.argwhere(np.all(walls == head_right, axis=-1, keepdims=False)).reshape((-1,))
    #     if len(t):
    #         indexes_right.append(t)
    #
    #     t = np.argwhere(np.all(walls == head_up, axis=-1, keepdims=False)).reshape((-1,))
    #     if len(t):
    #         indexes_up.append(t)
    #
    #     t = np.argwhere(np.all(walls == head_down, axis=-1, keepdims=False)).reshape((-1,))
    #     if len(t):
    #         indexes_down.append(t)

    indexes_down, indexes_up, indexes_left, indexes_right = \
        pool.map(process_heads, iter([(heads_down, walls), (heads_up, walls), (heads_left, walls), (heads_right, walls)]))

    idx_down = np.atleast_2d(
        walls[np.reshape(
            indexes_down,
            (-1,)).astype(int)
        ])[:, 0, None]
    idx_left = np.atleast_2d(
        walls[np.reshape(
            indexes_left,
            (-1,)).astype(int)
        ])[:, 0, None]
    idx_right = np.atleast_2d(
        walls[np.reshape(
            indexes_right,
            (-1,)).astype(int)
        ])[:, 0, None]

    idx_up = np.atleast_2d(
        walls[np.reshape(
            indexes_up,
            (-1,)).astype(int)
        ])[:, 0, None]

    if np.size(idx_up): mask[idx_up, BaseEnvironment.UP] = mask_with
    if np.size(idx_down): mask[idx_down, BaseEnvironment.DOWN] = mask_with
    if np.size(idx_right): mask[idx_right, BaseEnvironment.LEFT] = mask_with
    if np.size(idx_left): mask[idx_left, BaseEnvironment.RIGHT] = mask_with
    return tf.linalg.normalize(probs * mask, ord=1, axis=-1)[0]


