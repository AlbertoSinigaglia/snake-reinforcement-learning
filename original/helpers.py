from environments import *

def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            try:
                l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
            except:
                pass
        if hasattr(l,"bias_initializer"):
            try:
                l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
            except:
                pass
        if hasattr(l,"recurrent_initializer"):
            try:
                l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            except:
                pass

def re_normalize_possible_actions(state, probs, mask_with=0.):
    state = tf.argmax(state, axis=-1)
    board_size = state.shape[-1]

    indexes = tf.where(state == 2)
    mask = np.ones_like(probs)

    # [up, right, bottom, left]
    left_border = tf.where(indexes[:, 2] == 0)
    left_border = tf.concat((left_border, tf.ones_like(left_border) * 3), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, left_border, mask_with * tf.ones(left_border.shape[0]))

    right_border = tf.where(indexes[:, 2] == board_size - 1)
    right_border = tf.concat((right_border, tf.ones_like(right_border) * 1), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, right_border, mask_with * tf.ones(right_border.shape[0]))

    top_border = tf.where(indexes[:, 1] == board_size - 1)
    top_border = tf.concat((top_border, tf.ones_like(top_border) * 0), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, top_border, mask_with * tf.ones(top_border.shape[0]))

    bottom_border = tf.where(indexes[:, 1] == 0)
    bottom_border = tf.concat((bottom_border, tf.ones_like(bottom_border) * 2), axis=-1)
    mask = tf.tensor_scatter_nd_update(mask, bottom_border, mask_with * tf.ones(bottom_border.shape[0]))

    return tf.linalg.normalize(probs * mask, ord=1, axis=-1)[0]

