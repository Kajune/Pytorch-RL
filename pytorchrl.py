import random
import copy
import math
import numpy as np
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

def greedy_policy(policy_net, state, device, enable=None):
	with torch.no_grad():
		qvalues = policy_net(state)
		if enable is not None:
			qvalues *= torch.tensor(enable, device=device, dtype=torch.float)
		return qvalues.max(1)[1].view(1, 1)

def epsilon_greedy_policy(policy_net, state, n_actions, steps, eps_start, eps_end, eps_decay, device, enable=None):
	sample = random.random()
	eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps / eps_decay)
	if sample > eps_threshold:
		return greedy_policy(policy_net, state, device, enable)
	else:
		if enable is not None:
			return torch.tensor([[np.random.choice(np.where(enable)[0])]], device=device, dtype=torch.long)
		else:
			return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

class DQNAgent(object):
	def __init__(self, policy_net, target_net, device):
		self.device = device
		self.policy_net = policy_net.to(self.device)
		self.target_net = target_net.to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.RMSprop(self.policy_net.parameters())

	def optimize_model(self, memory, batch_size, gamma):
		if len(memory) < batch_size:
			return
		transitions = memory.sample(batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											  batch.next_state)), device=self.device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = self.policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(batch_size, device=self.device)
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * gamma) + reward_batch

		# Compute Huber loss
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()