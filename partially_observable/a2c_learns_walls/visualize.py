import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models import *
from environments import *
import numpy as np

n = 1
test_env = Walls9x9SnakeEnvironment(n, 2)

agent, _, _ = load_models(test_env, "models/Walls9x9SnakeEnvironment/9x9-1")


fig, ax = plt.subplots(1, 1, figsize=(5, 5))

def animate(_):
    probs = agent(test_env.to_state())
    actions = tf.random.categorical(tf.math.log(tf.stop_gradient(probs)), 1, dtype=tf.int32)
    ax.set_title(f"{test_env.move(actions).numpy().reshape(1)} - {actions.numpy().reshape(1)}")

    b = np.copy(test_env.boards[0]).astype(float)
    b = np.pad(b, test_env.mask_size)
    h = np.argwhere(b == test_env.HEAD)[0]
    b[(h[0] - test_env.mask_size):(h[0] + test_env.mask_size + 1),
    (h[1] - test_env.mask_size):(h[1] + test_env.mask_size + 1)] += 0.5
    b = b[test_env.mask_size:-test_env.mask_size, test_env.mask_size:-test_env.mask_size]
    image.set_data(b)
    return image


ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)


b = np.copy(test_env.boards[0]).astype(float)
b = np.pad(b, test_env.mask_size)
h = np.argwhere(b == test_env.HEAD)[0]
b[(h[0] - test_env.mask_size):(h[0] + test_env.mask_size + 1),
(h[1] - test_env.mask_size):(h[1] + test_env.mask_size + 1)] += 0.5
b = b[test_env.mask_size:-test_env.mask_size, test_env.mask_size:-test_env.mask_size]
image = ax.imshow(b, origin="lower")

anim = FuncAnimation(fig, animate, frames=1000, interval=90)
plt.show()
#anim.save('comparison.mp4', writer="ffmpeg")
