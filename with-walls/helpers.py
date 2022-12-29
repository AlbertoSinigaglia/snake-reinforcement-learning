from environments import *
from itertools import chain, islice, repeat

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
import numpy_indexed as npi

pool = Pool(cpu_count())
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

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

#
# def process_heads(params):
#     heads, walls = params
#     indexes = []
#     for head in heads:
#         t = np.argwhere(np.all(walls == head, axis=-1, keepdims=False)).reshape((-1,))
#         if len(t):
#             indexes.append(t)
#     return indexes
#
#
# def re_normalize_possible_actions_process(state, probs, mask_with=0.):
#     global pool
#     boards = tf.argmax(state, axis=-1)
#     heads = np.argwhere(boards == BaseEnvironment.HEAD)
#     walls = np.argwhere(boards == BaseEnvironment.WALL)
#     mask = np.ones_like(probs)
#
#     heads_up = np.copy(heads)
#     heads_down = np.copy(heads)
#     heads_left = np.copy(heads)
#     heads_right = np.copy(heads)
#     heads_up[:, 1] += 1
#     heads_down[:, 1] -= 1
#     heads_left[:, 2] += 1
#     heads_right[:, 2] -= 1
#
#     # MAKE THIS FASTER
#     indexes_down, indexes_up, indexes_left, indexes_right = \
#         pool.map(process_heads,
#                  iter([(heads_down, walls), (heads_up, walls), (heads_left, walls), (heads_right, walls)]))
#
#     idx_down = np.atleast_2d(
#         walls[np.reshape(
#             indexes_down,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#     idx_left = np.atleast_2d(
#         walls[np.reshape(
#             indexes_left,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#     idx_right = np.atleast_2d(
#         walls[np.reshape(
#             indexes_right,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#
#     idx_up = np.atleast_2d(
#         walls[np.reshape(
#             indexes_up,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#
#     if np.size(idx_up): mask[idx_up, BaseEnvironment.UP] = mask_with
#     if np.size(idx_down): mask[idx_down, BaseEnvironment.DOWN] = mask_with
#     if np.size(idx_right): mask[idx_right, BaseEnvironment.LEFT] = mask_with
#     if np.size(idx_left): mask[idx_left, BaseEnvironment.RIGHT] = mask_with
#     return tf.linalg.normalize(probs * mask, ord=1, axis=-1)[0]
#
#
#
#
#
# def process_head(els):
#     head, walls = els
#     return np.argwhere(np.all(walls == head, axis=-1, keepdims=False)).reshape((-1,))
#
#
# def process_heads(els):
#     heads, walls = els
#     return [np.argwhere(np.all(walls == head, axis=-1, keepdims=False)).reshape((-1,)) for head in heads]
#
#
# def re_normalize_possible_actions(state, probs, pool, mask_with=0.):
#     boards = tf.argmax(state, axis=-1)
#     heads = np.argwhere(boards == BaseEnvironment.HEAD)
#     walls = np.argwhere(boards == BaseEnvironment.WALL)
#     mask = np.ones_like(probs)
#
#     heads_up = np.copy(heads)
#     heads_down = np.copy(heads)
#     heads_left = np.copy(heads)
#     heads_right = np.copy(heads)
#     heads_up[:, 1] += 1
#     heads_down[:, 1] -= 1
#     heads_left[:, 2] += 1
#     heads_right[:, 2] -= 1
#
#     res = pool.map(process_head,
#                    zip(chain.from_iterable([heads_down, heads_up, heads_left, heads_right]), repeat(walls)))
#     indexes_down = res[:len(heads_down)]
#     indexes_up = res[len(heads_down):len(heads_down) + len(heads_up)]
#     indexes_left = res[len(heads_down) + len(heads_up):len(heads_down) + len(heads_up) + len(heads_left)]
#     indexes_right = res[len(heads_down) + len(heads_up) + len(heads_left):]
#
#     indexes_down = [el for el in indexes_down if len(el)]
#     indexes_up = [el for el in indexes_up if len(el)]
#     indexes_left = [el for el in indexes_left if len(el)]
#     indexes_right = [el for el in indexes_right if len(el)]
#
#     idx_down = np.atleast_2d(
#         walls[np.reshape(
#             indexes_down,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#     idx_left = np.atleast_2d(
#         walls[np.reshape(
#             indexes_left,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#     idx_right = np.atleast_2d(
#         walls[np.reshape(
#             indexes_right,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#
#     idx_up = np.atleast_2d(
#         walls[np.reshape(
#             indexes_up,
#             (-1,)).astype(int)
#         ])[:, 0, None]
#
#     if np.size(idx_up): mask[idx_up, BaseEnvironment.UP] = mask_with
#     if np.size(idx_down): mask[idx_down, BaseEnvironment.DOWN] = mask_with
#     if np.size(idx_right): mask[idx_right, BaseEnvironment.LEFT] = mask_with
#     if np.size(idx_left): mask[idx_left, BaseEnvironment.RIGHT] = mask_with
#     return tf.linalg.normalize(probs * mask, ord=1, axis=-1)[0]


def process_heads(els):
    heads, walls = els
    return [np.argwhere(np.all(walls == head, axis=-1, keepdims=False)).reshape((-1,)) for head in heads]


def re_normalize_possible_actions_chunk(state, probs, pool, mask_with=0., chunk_size=50):
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

    res = pool.map(process_heads,
                   zip(chunk(chain.from_iterable([heads_down, heads_up, heads_left, heads_right]),chunk_size), repeat(walls)))
    res = list(chain.from_iterable(res))
    indexes_down = res[:len(heads_down)]
    indexes_up = res[len(heads_down):len(heads_down) + len(heads_up)]
    indexes_left = res[len(heads_down) + len(heads_up):len(heads_down) + len(heads_up) + len(heads_left)]
    indexes_right = res[len(heads_down) + len(heads_up) + len(heads_left):]

    indexes_down = [el for el in indexes_down if len(el)]
    indexes_up = [el for el in indexes_up if len(el)]
    indexes_left = [el for el in indexes_left if len(el)]
    indexes_right = [el for el in indexes_right if len(el)]

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


def re_normalize_possible_actions_process_npi(state, probs, mask_with=0.):
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

    idx_down = walls[npi.indices(walls, heads_down, missing="ignore")][:,0]
    idx_up = walls[npi.indices(walls, heads_up, missing="ignore")][:,0]
    idx_left = walls[npi.indices(walls, heads_left, missing="ignore")][:,0]
    idx_right = walls[npi.indices(walls, heads_right, missing="ignore")][:,0]

    if np.size(idx_up): mask[idx_up, BaseEnvironment.UP] = mask_with
    if np.size(idx_down): mask[idx_down, BaseEnvironment.DOWN] = mask_with
    if np.size(idx_right): mask[idx_right, BaseEnvironment.LEFT] = mask_with
    if np.size(idx_left): mask[idx_left, BaseEnvironment.RIGHT] = mask_with
    return tf.linalg.normalize(probs * mask, ord=1, axis=-1)[0]


