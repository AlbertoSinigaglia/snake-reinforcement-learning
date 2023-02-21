import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models import *
import environments_masked
import environments
import numpy as np

n = 1
test_env_f = environments.OriginalSnakeEnvironment(n, 10)
test_env_1 = environments_masked.OriginalSnakeEnvironment(n, 10, 1)
test_env_2 = environments_masked.OriginalSnakeEnvironment(n, 10, 2)

print("loading f")
agent_f, _, _ = load_models_big(test_env_f, "models/OriginalSnakeEnvironment/10x10")
print("loading 1")
agent_1, _, _ = load_models(test_env_1, "models/OriginalSnakeEnvironment/10x10-1")
print("loading 2")
agent_2, _, _ = load_models(test_env_2, "models/OriginalSnakeEnvironment/10x10-2")
print("end")


fig, axs = plt.subplots(1, 3, figsize=(10, 5))
images = []


def animate(_):
    probs = agent_f(test_env_f.to_state())
    actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)
    test_env_f.move(actions)

    probs = agent_1(test_env_1.to_state())
    actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)
    test_env_1.move(actions)

    probs = agent_2(test_env_2.to_state())
    actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)
    test_env_2.move(actions)

    b = np.copy(test_env_f.boards[0]).astype(float)
    images[0].set_data(b)

    b = np.copy(test_env_1.boards[0]).astype(float)
    b = np.pad(b, test_env_1.mask_size)
    h = np.argwhere(b == test_env_1.HEAD)[0]
    b[(h[0] - test_env_1.mask_size):(h[0] + test_env_1.mask_size + 1),
    (h[1] - test_env_1.mask_size):(h[1] + test_env_1.mask_size + 1)] += 0.5
    b = b[test_env_1.mask_size:-test_env_1.mask_size, test_env_1.mask_size:-test_env_1.mask_size]
    images[1].set_data(b)

    b = np.copy(test_env_2.boards[0]).astype(float)
    b = np.pad(b, test_env_2.mask_size)
    h = np.argwhere(b == test_env_2.HEAD)[0]
    b[(h[0] - test_env_2.mask_size):(h[0] + test_env_2.mask_size + 1),
    (h[1] - test_env_2.mask_size):(h[1] + test_env_2.mask_size + 1)] += 0.5
    b = b[test_env_2.mask_size:-test_env_2.mask_size, test_env_2.mask_size:-test_env_2.mask_size]
    images[2].set_data(b)
    return images


for ax in axs:
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

b = np.copy(test_env_f.boards[0]).astype(float)
images.append(axs[0].imshow(b, origin="lower"))

b = np.copy(test_env_1.boards[0]).astype(float)
b = np.pad(b, test_env_1.mask_size)
h = np.argwhere(b == test_env_1.HEAD)[0]
b[(h[0] - test_env_1.mask_size):(h[0] + test_env_1.mask_size + 1),
(h[1] - test_env_1.mask_size):(h[1] + test_env_1.mask_size + 1)] += 0.5
b = b[test_env_1.mask_size:-test_env_1.mask_size, test_env_1.mask_size:-test_env_1.mask_size]
images.append(axs[1].imshow(b, origin="lower"))

b = np.copy(test_env_2.boards[0]).astype(float)
b = np.pad(b, test_env_2.mask_size)
h = np.argwhere(b == test_env_2.HEAD)[0]
b[(h[0] - test_env_2.mask_size):(h[0] + test_env_2.mask_size + 1),
(h[1] - test_env_2.mask_size):(h[1] + test_env_2.mask_size + 1)] += 0.5
b = b[test_env_2.mask_size:-test_env_2.mask_size, test_env_2.mask_size:-test_env_2.mask_size]
images.append(axs[2].imshow(b, origin="lower"))

anim = FuncAnimation(fig, animate, frames=1000, interval=90)
plt.show()
#anim.save('comparison.mp4', writer="ffmpeg")
