from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from gymPacMan import gymPacMan_parallel_env

torch.autograd.set_detect_anomaly(True)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# possible options: env = gymPacMan_parallel_env(layout_file = f'{file_path}/layouts/smallCapture.lay', display = False, length = 800, reward_forLegalAction= True, defenceReward = True, random_layout = True, enemieName = 'randomTeam', self_play = False)
env = gymPacMan_parallel_env(layout_file='./layouts/tinyCapture.lay', display=False, length=500)
obs, _ = env.reset()

# Start a new wandb run to track this script

wandb.login(key = '', relogin = True)
name_experiment =''
wandb.init(project="", id = f"{name_experiment}__{datetime.now().strftime('%Y%m%d_%H%M%S')}")


class AgentQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=64):
        super(AgentQNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        # Pass through convolutional layers
        x = self.conv1(obs)
        x = self.conv2(x)

        # Flatten the output
        x = self.flatten(x)

        # Pass through fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        # Uncomment the following line if you want to include the second fully connected layer

        # Output Q-values
        q_values = self.fc2(x)
        return q_values


class SimpleQMixer(nn.Module):
    def __init__(self, n_agents, state_shape):
        """
        Simple QMIX Mixing Network with ReLU activation.
        Args:
            n_agents (int): Number of agents.
            state_shape (int): Dimension of the global state.
        """
        super(SimpleQMixer, self).__init__()


    def forward(self, agent_qs, states):
        """
        Forward pass for the mixing network with ReLU activation.
        Args:
            agent_qs (torch.Tensor): Tensor of shape [batch_size, n_agents] containing Q-values for each agent.
            states (torch.Tensor): Tensor of shape [batch_size, state_shape] representing the global state.
        Returns:
            torch.Tensor: Global Q-value of shape [batch_size, 1].
        """

        return 0


def compute_td_loss(agent_q_networks, mixing_network, target_q_networks, batch, weights=None, gamma=0.99, lambda_=0.1):
    """
    Computes the TD loss for QMix training using the Huber loss.

    Args:
        agent_q_networks (list): List of Q-networks for each agent.
        mixing_network (SimpleQMixer): The mixing network to compute global Q-values.
        target_q_networks (list): List of target Q-networks for each agent.
        batch (tuple): A batch of experiences (states, actions, rewards, next_states, dones).
        weights (torch.Tensor): Importance sampling weights (optional).
        gamma (float): Discount factor for future rewards.
        lambda_ (float): Regularization factor for stability.

    Returns:
        torch.Tensor: Total loss for training.
    """
    states, actions, rewards, next_states, dones = batch

    # Convert to tensors and move to device
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Current Q-values for each agent
    agent_q_values = []
    for agent_index, q_net in enumerate(agent_q_networks):
        q_vals = q_net(states[:, agent_index, :, :, :])  # Get Q-values for each agent
        agent_q_values.append(
            q_vals.gather(dim=1, index=actions[:, agent_index].unsqueeze(1)))  # Select Q-value for taken action
    agent_q_values = torch.cat(agent_q_values, dim=1)  # Shape: (batch_size, n_agents)

    # Target Q-values using Double DQN
    with torch.no_grad():
        # Get actions from current Q-networks
        next_agent_q_values = []
        for agent_index, (q_net, target_net) in enumerate(zip(agent_q_networks, target_q_networks)):
            next_q_vals = q_net(next_states[:, agent_index, :, :, :])  # Get Q-values from current network
            max_next_actions = next_q_vals.argmax(dim=1, keepdim=True)  # Greedy actions
            target_q_vals = target_net(next_states[:, agent_index, :, :, :])  # Get Q-values from target network\
            max_next_q_vals = target_q_vals.gather(1, max_next_actions)
            done_mask = dones[:, 0, 0].unsqueeze(1)
            filtered_target_q_vals = max_next_q_vals * (1 - done_mask)

            next_agent_q_values.append(filtered_target_q_vals)  # Use target Q-values for selected actions
        next_agent_q_values = torch.cat(next_agent_q_values, dim=1)  # Shape: (batch_size, n_agents)

    # Independent Q-learning target for each agent (all members of the blue team receive the same reward)
    target_q = rewards[:, 0, 0].unsqueeze(1) + gamma * next_agent_q_values

    # Compute Huber loss, try also with MSE loss
    loss_fn = torch.nn.HuberLoss()

    loss_agent1 = loss_fn(agent_q_values[:, 0], target_q[:, 0])
    loss_agent2 = loss_fn(agent_q_values[:, 1], target_q[:, 1])

    return loss_agent1, loss_agent2


from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # Restructure the batch into separate arrays for states, actions, rewards, next_states, and dones
        states = np.array([exp[0].cpu().numpy() for exp in experiences], dtype=np.float32)
        actions = np.array([exp[1] for exp in experiences], dtype=np.int64)
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3].cpu().numpy() for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


def epsilon_greedy_action(agent_q_network, state, epsilon, legal_actions):
    if random.random() < epsilon:
        # Explore: take a random action
        action = random.choice(legal_actions)
    else:
        state = torch.unsqueeze(state.clone().detach(), 0).to(device)
        q_values = agent_q_network(state).cpu().detach().numpy()
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

    return action


def update_target_network(agent_q_networks, target_q_networks):
    for target, source in zip(target_q_networks, agent_q_networks):
        target.load_state_dict(source.state_dict())


def train_qmix(env, agent_q_networks, mixing_network, target_q_networks, replay_buffer, n_episodes=10000,
               batch_size=32, gamma=0.9):
    optimizer = optim.Adam([param for net in agent_q_networks for param in net.parameters()], lr=0.0001)

    epsilon = .75 # Initial exploration probability
    epsilon_min = 0.1
    epsilon_decay = 0.99
    target_update_frequency = 10
    steps_counter = 0
    legal_actions = [0, 1, 2, 3, 4]
    agent_indexes = [1, 3]

    for episode in range(n_episodes):
        done = {agent_id: False for agent_id in agent_indexes}
        env.reset()
        blue_player1_reward = 0
        blue_player2_reward = 0
        score = 0
        while not all(done.values()):
            actions = [-1 for _, _ in enumerate(env.agents)]
            states = []
            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                state = torch.tensor(obs_agent, dtype=torch.float32).to(device)
                states.append(state)
                action = epsilon_greedy_action(agent_q_networks[i], state, epsilon, legal_actions)
                actions[agent_index] = action



            next_states, rewards, terminations, info = env.step(actions)
            score -= info["score_change"]
            done = {key: value for key, value in terminations.items() if key in agent_indexes}
            blue_player1_reward += rewards[1]
            blue_player2_reward += rewards[3]

            next_states_converted = []
            rewards_converted = []
            terminations_converted = []
            actions_converted = []

            for index in agent_indexes:
                next_states_converted.append(list(next_states.values())[index])
                rewards_converted.append(rewards[index])
                terminations_converted.append(terminations[index])
                actions_converted.append(actions[index])

            next_states_converted = torch.stack(next_states_converted)
            states_converted = torch.stack(states)
            rewards_converted = [rewards_converted]
            terminations_converted = [terminations_converted]
            replay_buffer.add(
                (states_converted, actions_converted, rewards_converted, next_states_converted, terminations_converted))

            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss1, loss2 = compute_td_loss(agent_q_networks, mixing_network, target_q_networks, batch,
                                               gamma=gamma)
                wandb.log({"loss1": loss1})
                wandb.log({"loss2": loss2})

                # Zero gradients for all optimizers
                optimizer.zero_grad()

                # Backpropagate once for all losses
                loss1.backward(retain_graph=True)
                loss2.backward()

                # Update weights
                optimizer.step()

        steps_counter += 1
        wandb.log({"epsilon": epsilon})
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        wandb.log({"blue_player1_reward": blue_player1_reward})
        wandb.log({"blue_player2_reward": blue_player2_reward})
        wandb.log({"episode": episode})
        wandb.log({'score': score})




        if (episode + 1) % target_update_frequency == 0:
            update_target_network(agent_q_networks, target_q_networks)


n_agents = int(len(env.agents) / 2)
action_dim_individual_agent = 5  # North, South, East, West, Stop

obs_individual_agent = env.get_Observation(0)
obs_shape = obs_individual_agent.shape

agent_q_networks = [AgentQNetwork(obs_shape=obs_shape, action_dim=action_dim_individual_agent).to(device) for _ in
                    range(n_agents)]
target_q_networks = [AgentQNetwork(obs_shape=obs_shape, action_dim=action_dim_individual_agent).to(device) for _ in
                     range(n_agents)]

# Initialize target Q-networks with the same weights as the main Q-networks
update_target_network(agent_q_networks, target_q_networks)

# Initialize the replay buffer
replay_buffer = ReplayBuffer()

# Train QMix in the gymPacMan environment

global_state_shape = [len(env.agents)] + list(obs_shape)
mixing_network = SimpleQMixer(n_agents=n_agents, state_shape=global_state_shape).to(device)

train_qmix(env, agent_q_networks, mixing_network, target_q_networks, replay_buffer)

