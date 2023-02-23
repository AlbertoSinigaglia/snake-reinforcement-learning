import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models import *
from environments import *
def get_env(n=1000):
    return Walls25x25SnakeEnvironment(n)
test_env = get_env(3)


class GreedyPolicy:
    def sample(self, q_values):
        return np.argmax(q_values, axis=-1, keepdims=True)
policy = GreedyPolicy()

MODELS_PREFIX = f"models/{type(test_env).__name__}/{test_env.board_size}x{test_env.board_size}"
qfunction, _ = load_models(test_env, MODELS_PREFIX)




fig, axs = plt.subplots(1,3, figsize=(10,5))
images = []
def animate(_):
    actions = policy.sample(qfunction(test_env.to_state()))
    test_env.move(actions)
    for board, image in zip(test_env.boards, images):
        image.set_data(board)
    return images
for b, ax in zip(test_env.boards, axs.flatten()):
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    images.append(ax.imshow(b, origin="lower"))
anim = FuncAnimation(fig, animate, interval=60)
plt.show()
