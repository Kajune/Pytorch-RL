import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import pytorchrl as rl
import cartpole

#env = gym.make('CartPole-v0').unwrapped
env = cartpole.CartPoleEnv()

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation=Image.CUBIC),
					T.ToTensor()])


def get_cart_location(screen_width):
	world_width = env.x_threshold * 2
	scale = screen_width / world_width
	return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
	# Returned screen requested by gym is 400x600x3, but is sometimes larger
	# such as 800x1200x3. Transpose it into torch order (CHW).
	screen = env.render(mode='rgb_array').transpose((2, 0, 1))
	# Cart is in the lower half, so strip off the top and bottom of the screen
	_, screen_height, screen_width = screen.shape
	screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
	view_width = int(screen_width * 0.6)
	cart_location = get_cart_location(screen_width)
	if cart_location < view_width // 2:
		slice_range = slice(view_width)
	elif cart_location > (screen_width - view_width // 2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width // 2,
							cart_location + view_width // 2)
	# Strip off the edges, so that we have a square image centered on a cart
	screen = screen[:, :, slice_range]
	# Convert to float, rescale, convert to torch tensor
	# (this doesn't require a copy)
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
		   interpolation='none')
plt.title('Example extracted screen')
plt.show()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions)
target_net = DQN(screen_height, screen_width, n_actions)
agent = rl.DQNAgent(policy_net, target_net, device)

memory = rl.ReplayMemory(10000)


steps_done = 0
episode_durations = []

def plot_durations():
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated

num_episodes = 500
for i_episode in range(num_episodes):
	# Initialize the environment and state
	env.reset()
	last_screen = get_screen()
	current_screen = get_screen()
	state = current_screen - last_screen
	for t in count():
		# Select and perform an action
		action = rl.epsilon_greedy_policy(policy_net, state, n_actions, steps_done, EPS_START, EPS_END, EPS_DECAY, device)
		steps_done += 1
		_, reward, done, _ = env.step(action.item())
		reward = torch.tensor([reward], device=device)

		# Observe new state
		last_screen = current_screen
		current_screen = get_screen()
		if not done:
			next_state = current_screen - last_screen
		else:
			next_state = None

		# Store the transition in memory
		memory.push(state, action, next_state, reward)

		# Move to the next state
		state = next_state

		# Perform one step of the optimization (on the target network)
		agent.optimize_model(memory, BATCH_SIZE, GAMMA)
		if done:
			episode_durations.append(t + 1)
			plot_durations()
			break
	# Update the target network, copying all weights and biases in DQN
	if i_episode % TARGET_UPDATE == 0:
		target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()