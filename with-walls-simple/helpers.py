

from environments import *

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

pool = Pool(cpu_count())

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



def re_normalize_possible_actions(boards, probs, mask_with=0.):
    heads = np.argwhere(boards == BaseEnvironment.HEAD)
    walls = np.argwhere(boards == BaseEnvironment.WALL)
    mask = np.ones_like(probs)

    heads_up = np.copy(heads)
    heads_down = np.copy(heads)
    heads_left = np.copy(heads)
    heads_right = np.copy(heads)

    heads_up[:, 1] += 1
    heads_down[:, 1] -= 1
    heads_right[:, 2] += 1
    heads_left[:, 2] -= 1

    powers = np.max(walls)**np.arange(walls.shape[1])

    walls_code = np.sum(walls*powers, axis=-1)
    heads_up_code = np.sum(heads_up*powers, axis=-1)
    heads_down_code = np.sum(heads_down*powers, axis=-1)
    heads_left_code = np.sum(heads_left*powers, axis=-1)
    heads_right_code = np.sum(heads_right*powers, axis=-1)

    idx_up = heads_up[np.isin(heads_up_code, walls_code)][:,0]
    idx_down = heads_up[np.isin(heads_down_code, walls_code)][:,0]
    idx_right = heads_up[np.isin(heads_right_code, walls_code)][:,0]
    idx_left = heads_up[np.isin(heads_left_code, walls_code)][:,0]

    if np.size(idx_up): mask[idx_up, BaseEnvironment.UP] = mask_with
    if np.size(idx_down): mask[idx_down, BaseEnvironment.DOWN] = mask_with
    if np.size(idx_right): mask[idx_right, BaseEnvironment.RIGHT] = mask_with
    if np.size(idx_left): mask[idx_left, BaseEnvironment.LEFT] = mask_with

    return tf.linalg.normalize(probs * mask, ord=1, axis=-1)[0]

