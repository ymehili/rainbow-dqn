import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from PIL import Image
import time
from gymnasium.wrappers import AtariPreprocessing
import os

# --- Hyperparameters ---
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
PRIORITY_EPS = 1e-6
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_END = 1.0
PRIORITY_BETA_DECAY = 0.99995
N_ATOM = 51
V_MIN = -10
V_MAX = 10
DZ = (V_MAX - V_MIN) / (N_ATOM - 1)
LEARN_STEPS = 5

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessing(env)
state_shape = env.observation_space.shape
n_actions = env.action_space.n
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


def stack_frames(stacked_frames, frame, is_new_episode):
    """Stack frames into a state representation."""
    frame = frame / 255.0
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
        for _ in range(4):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    stacked_state = np.stack(list(stacked_frames), axis=0)
    return stacked_frames, stacked_state


class DuelingQNetwork(nn.Module):
    def __init__(self, state_shape, n_actions, n_atoms, seed=0):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.z = torch.linspace(V_MIN, V_MAX, n_atoms).to(device)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_adv1 = nn.Linear(64 * 7 * 7, 512)
        self.fc_adv2 = nn.Linear(512, n_actions * n_atoms)

        self.fc_val1 = nn.Linear(64 * 7 * 7, 512)
        self.fc_val2 = nn.Linear(512, n_atoms)



    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)
        adv = adv.view(-1, self.n_actions, self.n_atoms)

        val = F.relu(self.fc_val1(x))
        val = self.fc_val2(val)
        val = val.view(-1, 1, self.n_atoms)

        q_dist = val + adv - adv.mean(dim=1, keepdim=True)
        q_probs = F.softmax(q_dist, dim=2)
        return q_probs

    def get_q_values(self, state):
        """Compute the Q values (expected returns) from the distribution."""
        q_dist = self.forward(state)
        q_values = torch.sum(q_dist * self.z, dim=2)
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=0):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done, error=None):
        """Add a new experience to memory, using TD error as initial priority."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority if error is None else error)


    def sample(self, beta):
        """Sample a batch of experiences from memory, using prioritized sampling."""
        probs = np.array(self.priorities) ** PRIORITY_ALPHA
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs, replace=False)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(device).view(-1, 1)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, indices)


    def update_priorities(self, indices, errors):
        """Update priorities of sampled experiences based on TD errors."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + PRIORITY_EPS

    def __len__(self):
        return len(self.buffer)



class RainbowAgent:
    def __init__(self, state_shape, n_actions, n_atoms, seed=0):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.seed = random.seed(seed)

        self.qnetwork_local = DuelingQNetwork(state_shape, n_actions, n_atoms, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_shape, n_actions, n_atoms, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        self.beta = PRIORITY_BETA_START
        self.learn_counter = 0


    def step(self, state, action, reward, next_state, done):
        q_values_local = self.qnetwork_local.get_q_values(torch.tensor(state).unsqueeze(0).float().to(device))
        initial_error = abs(reward - q_values_local[0, action].item())

        self.memory.add(state, action, reward, next_state, done, initial_error)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                self.learn_counter = (self.learn_counter + 1) % LEARN_STEPS
                if self.learn_counter == 0:
                    experiences = self.memory.sample(self.beta)
                    self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local.get_q_values(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, weights, indices = experiences

        with torch.no_grad():
            next_q_dist_target = self.qnetwork_target(next_states)

            next_q_values_local = self.qnetwork_local.get_q_values(next_states)
            best_actions_local = next_q_values_local.argmax(dim=1)

            next_q_dist_target_best = next_q_dist_target[torch.arange(BATCH_SIZE), best_actions_local]

            tz = (rewards + (1 - dones) * GAMMA * self.qnetwork_target.z).clamp(V_MIN, V_MAX)
            bj = (tz - V_MIN) / DZ

            l = bj.floor().long()
            u = bj.ceil().long()

            l[(u > 0) * (l == u)] -= 1
            u[(l < (N_ATOM - 1)) * (l == u)] += 1

            m = states.new_zeros(BATCH_SIZE, N_ATOM)
            offset = torch.linspace(0, (BATCH_SIZE - 1) * N_ATOM, BATCH_SIZE).long()\
                            .unsqueeze(1).expand(BATCH_SIZE, N_ATOM).to(device)


            m.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist_target_best * (u.float() - bj)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist_target_best * (bj - l.float())).view(-1))

        q_dist_local = self.qnetwork_local(states)
        q_dist_local_taken = q_dist_local[torch.arange(BATCH_SIZE), actions.squeeze()]

        loss = -torch.sum(weights * m * torch.log(q_dist_local_taken + 1e-8), dim=1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        with torch.no_grad():
            errors = -torch.sum(m * torch.log(q_dist_local_taken + 1e-8), dim=1).cpu().numpy() # TD errors
            self.memory.update_priorities(indices, errors)

        self.beta = min(self.beta * PRIORITY_BETA_DECAY + (1-PRIORITY_BETA_DECAY) * PRIORITY_BETA_END , PRIORITY_BETA_END)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# --- Training Loop ---
def train_rainbow(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    agent = RainbowAgent(state_shape=(4, 84, 84), n_actions=n_actions, n_atoms=N_ATOM)
    start_time = time.time()

    for i_episode in range(1, n_episodes+100):
        if i_episode % 100 == 0 or i_episode > 1000:
            current_env = gym.make("ALE/MsPacman-v5", render_mode="human", frameskip=1)
        else:
            current_env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=1)

        current_env = AtariPreprocessing(current_env)

        state, info = current_env.reset()
        stacked_frames, stacked_state = stack_frames(None, state, True)  # First state
        score = 0

        for t in range(max_t):
            action = agent.act(stacked_state, eps)
            next_state, reward, terminated, truncated, _ = current_env.step(action)
            done = terminated or truncated
            stacked_frames, stacked_next_state = stack_frames(stacked_frames, next_state, False)
            agent.step(stacked_state, action, reward, stacked_next_state, done)
            stacked_state = stacked_next_state
            score += reward
            if done:
                break
        current_env.close()
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        elapsed_time = time.time() - start_time
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {elapsed_time:.2f}s',
              end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {elapsed_time:.2f}s')
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_{i_episode}.pth')
    return scores

scores = train_rainbow()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()
