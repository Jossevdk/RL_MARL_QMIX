
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
env = gymPacMan_parallel_env(layout_file='./layouts/tinyCapture.lay', display=False, length=300, enemieName='MCTSTeam.py')
obs, _ = env.reset()

# Start a new wandb run to track this script

wandb.login(key = '', relogin = True)
name_experiment =''

layout = ['bloxCapture.lay','defence.lay','smallCapture.lay','tinyCapture.lay','verdedeging2.lay','win.lay','win2.lay']
for lay in layout:
    env = gymPacMan_parallel_env(layout_file=f'./layouts/{lay}', display=False, length=300, enemieName='MCTSTeam.py')
    print(env.get_Observation(0).shape)
# **Section 1: QMix Implementation**

##Let’s begin!

## 1.1   Agent Q-Network Implementation
class AgentQNetwork(nn.Module):
    def __init__(self,obs_shape, action_dim,hidden_dim=64):
        super(AgentQNetwork, self).__init__()
        self.c1 = nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(32, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(4*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)


    def forward(self, obs):
        x = torch.relu(self.c1(obs))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = torch.relu(self.c4(x))
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
import torch.nn.functional as F

class AgentQNetwork2(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=64):
        super(AgentQNetwork2, self).__init__()
        
        self.c1 = nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1, padding='same')
        self.c2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32, action_dim)
        
        self.activation = nn.SiLU()  # Use SiLU as the activation function

    def forward(self, obs):
        x = self.activation(self.bn1(self.c1(obs)))
        x = self.activation(self.bn2(self.c2(x)))
        x = self.activation(self.bn3(self.c3(x)))
        
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

class CNNTransformerQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, d_model=32, n_heads=2, n_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(2)  # (batch, channels, H*W)
        self.token_proj = nn.Linear(32, d_model)  # Project to transformer tokens
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads),
            num_layers=n_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, action_dim),
            nn.SiLU()
        )

    def forward(self, obs):
        x = self.cnn(obs)
        batch_size, channels, height, width = x.size()
        x = self.flatten(x).permute(0, 2, 1)  # Tokens: (batch, H*W, channels)
        x = self.token_proj(x)  # Project tokens
        x = self.transformer(x)  # Apply transformer
        x = x.mean(dim=1)  # Pool tokens
        q_values = self.fc(x)
        return q_values

# Define the CNN-LSTM-Q Network

#TODO: include sequences as input
#TODO: use padding for the batch size


class CNNLSTMQNetwork_hidden(nn.Module):
    def __init__(self, embed_dim, obs_shape, action_space):
        super(CNNLSTMQNetwork_hidden, self).__init__()
        self.embed_dim = embed_dim

        # Shared encoder for the agent's state
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to handle varying input sizes
        )

        # Fully connected layer to project encoded state to embed_dim
        self.fc = nn.Linear(128 * 1 * 1, embed_dim)

        # LSTM layer to model temporal dependencies
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)

        # Final layer to compute Q-values
        self.final_layer = nn.Sequential(
            nn.Linear(embed_dim, action_space),
            nn.SiLU()  # Swish activation
        )

    def forward(self, state, hidden):
        # Process single state using the shared encoder
        cnn_features = []
        for t in range(5):
            
            encoded_state = self.shared_encoder(state[:, t, :, :, :])  # Encode the state
            encoded_state = encoded_state.view(encoded_state.size(0), -1)  # Flatten to [batch_size, 128 * 1 * 1]
            projected_state = self.fc(encoded_state).unsqueeze(1)  # Project to embed_dim and add sequence dimension
            cnn_features.append(projected_state)
            
        cnn_features = torch.cat(cnn_features, dim=1)  # Shape: (batch_size, seq_len, cnn_output_size)
        # Apply LSTM to model temporal dependencies
        hx, cx = hidden[0, :, :, :], hidden[1, :, :, :]
        lstm_out, (hx, cx) = self.lstm(cnn_features, (hx, cx))
        hidden_new = torch.stack([hx.detach(), cx.detach()], dim=0)
        # Use the output of the LSTM for the final layer
        q_values = self.final_layer(lstm_out[:, -1, :])  # Shape: [batch_size, action_space]

        return q_values, hidden_new

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.embed_dim),
                torch.zeros(1, batch_size, self.embed_dim))




import torch.nn.functional as F
class SimpleQMixer(nn.Module):
    def __init__(self, n_agents, state_shape):
        super(SimpleQMixer, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape
        print(state_shape)
        # Much simpler state processing
        self.state_conv = nn.Sequential(
            nn.Conv3d(state_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * state_shape[1] * state_shape[2] * state_shape[3],  n_agents)  # Output: (batch_size, 32 * n_agents)
        )
        # Initialize weights positively
        for m in self.state_conv:
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.9, 1.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        # Single layer mixing network
        self.mixing_layer = nn.Linear(n_agents , 1, bias=False)

        # Initialize close to equal weights
        nn.init.uniform_(self.mixing_layer.weight, 0.9, 1.1)

    def forward(self, agent_qs, states):

        bs = agent_qs.size(0)
        weights = F.softplus(self.state_conv(states))


        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        weighted_qs = agent_qs * weights.unsqueeze(1)

        q_tot = self.mixing_layer(weighted_qs.squeeze(1))
        return q_tot.view(bs, -1, 1)



import torch
import torch.nn as nn

class AttentionMixer(nn.Module):
    def __init__(self, n_agents, state_shape=(2, 8, 10, 20), embed_dim=32, n_heads=4):
        super(AttentionMixer, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape[1:]  # Remove agent dimension
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Shared Conv2D encoder for each agent
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(self.state_shape[0], 16, kernel_size=3, stride=1, padding='same'),
            nn.SiLU(),  # Swish activation
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.SiLU(),  # Swish activation
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.SiLU(),  # Swish activation
            nn.AdaptiveAvgPool2d(1)
        )

        # Fully connected layer to project to embed_dim
        self.fc = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.SiLU()  # Swish activation
        )

        for m in self.shared_encoder:
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, 0.9, 1.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.9, 1.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        # Multi-head attention for inter-agent interaction
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        # Hypernetwork for mixing weights and biases
        self.hyper_w = nn.Sequential(
            nn.Linear(embed_dim * n_agents, embed_dim * embed_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(embed_dim * embed_dim, n_agents * embed_dim)
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(embed_dim * n_agents, embed_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(embed_dim, embed_dim)
        )

        # Final layer to compute Q_tot
        self.final_layer = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.SiLU()  # Swish activation
        )

        for m in self.hyper_w:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.9, 1.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        for m in self.hyper_b:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.9, 1.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        for m in self.final_layer:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.9, 1.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)


    def forward(self, agent_qs, states):
        batch_size = states.size(0)

        # Process each agent's state individually using the shared encoder
        agent_features = []
        for i in range(self.n_agents):
            agent_state = states[:, i, :, :, :]  # Shape: [batch_size, 8, 10, 20]
            encoded_state = self.shared_encoder(agent_state)  # Encode individual agent
            encoded_state = encoded_state.view(batch_size, -1)  # Flatten to [batch_size, 128]
            projected_state = self.fc(encoded_state)  # Project to embed_dim
            agent_features.append(projected_state)

        # Stack agent features: Shape [batch_size, n_agents, embed_dim]
        agent_features = torch.stack(agent_features, dim=1)

        # Apply attention to model inter-agent interaction
        attention_out, _ = self.attention(agent_features, agent_features, agent_features)

        # Flatten attention output for hypernetwork
        attention_out_flat = attention_out.reshape(batch_size, -1)  # Shape: [batch_size, n_agents * embed_dim]

        # Compute hypernetwork weights and biases
        hyper_w = F.softplus(self.hyper_w(attention_out_flat).view(batch_size, self.n_agents, self.embed_dim))
        hyper_b = self.hyper_b(attention_out_flat).view(batch_size, self.embed_dim)

        # Combine agent Q-values with weights and biases
        agent_qs = agent_qs.view(batch_size, self.n_agents, 1)  # Shape: [batch_size, n_agents, 1]
        weighted_qs = torch.bmm(agent_qs.transpose(1, 2), hyper_w).squeeze(1) + hyper_b

        # Final projection to Q_tot
        q_tot = self.final_layer(weighted_qs).unsqueeze(1)
        return q_tot



def compute_td_loss(agent_q_networks, mixing_network, target_q_networks, batch, weights = None, gamma=0.99, lambda_=0.1):
    states, actions, rewards, next_states, dones, hidden_states, next_hidden_states = batch
    
     # Get current Q-values for each agent
    agent_qs = []
    for i, agent_q_network in enumerate(agent_q_networks):
        agent_q, _ = agent_q_network(states[:,:, i, :, :, :], hidden_states[i,:, :,:,  :])  # Shape: (batch_size, num_actions)
       
        agent_q = agent_q.gather(dim=1, index=actions[:,-1,i].unsqueeze(1))  # Select action Q-values
        agent_qs.append(agent_q)

    agent_qs = torch.cat(agent_qs, dim=1)  # Shape: (batch_size, n_agents)
    # Get target Q-values for each agent using Double DQN
    with torch.no_grad():
        target_agent_qs = []
        for i, (agent_q_network, target_q_network) in enumerate(
            zip(agent_q_networks, target_q_networks)
        ):
            # Double DQN: Get greedy actions from the current network

            next_q_values, _ = agent_q_network(next_states[:,:, i, :, :, :], next_hidden_states[i,:, :, :, :])  # Shape: (batch_size, num_actions)
            greedy_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
            # Get target Q-values using the target network
            target_q_values, _ = target_q_network(next_states[:,:, i, :, :, :], next_hidden_states[i,:, :, :, :])  # Shape: (batch_size, num_actions)
            target_q = target_q_values.gather(1, greedy_actions)
           
            done_mask = dones[-1, i, :].unsqueeze(1)
            filtered_target_q = target_q * (1 - done_mask)
            target_agent_qs.append(filtered_target_q)
        
        target_agent_qs = torch.cat(target_agent_qs, dim=1)  # Shape: (batch_size, n_agents)
        
        # Compute the total target Q-value using the mixing network
        target_q_tot = mixing_network(target_agent_qs, next_states[:, -1, :, :, :, :].contiguous()).squeeze(1)   # Shape: (batch_size, 1)
    target_q_tot = rewards[:, -1, :].sum(dim=1).unsqueeze(1) + gamma * target_q_tot


    # Compute the global Q-value using the mixing network
    q_tot = mixing_network(agent_qs, states[:, -1, :, :, :, :].contiguous()).squeeze(1)  # Shape: (batch_size, 1)
    
    # Compute Huber loss
    td_loss = F.huber_loss(target_q_tot, q_tot,  reduction="none")

    # Apply weights if prioritized experience replay is used
    if weights is not None:
        td_loss *= weights

    # Return the mean loss
    total_loss = td_loss.mean()
    return total_loss
## 1.4 Training the QMix Algorithm

from collections import deque
import random

from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # Stack the experiences into tensors directly on the GPU
        
        states = torch.stack([torch.stack(exp[0]).to('cuda') for exp in experiences])
        actions = torch.stack([torch.tensor(exp[1]).to('cuda') for exp in experiences])
        rewards = torch.stack([torch.tensor(exp[2]).to('cuda') for exp in experiences])
        next_states = torch.stack([torch.stack(exp[3]).to('cuda') for exp in experiences])
        terminations = torch.stack([torch.tensor(exp[4], dtype=torch.int).to('cuda') for exp in experiences])
        hidden_states = torch.cat([exp[5] for exp in experiences], dim = 3)
        next_hidden_states = torch.cat([exp[6] for exp in experiences], dim = 3)
        

        return (states, actions, rewards, next_states, terminations, hidden_states, next_hidden_states)

    def size(self):
        return len(self.buffer)
class ReplayBuffer_sequence:
    def __init__(self, buffer_size=10000, sequence_length=5):
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length

    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sequences = []
        for _ in range(batch_size):
            start_index = random.randint(0, len(self.buffer) - self.sequence_length)
            sequence = [self.buffer[i] for i in range(start_index, start_index + self.sequence_length)]
            sequences.append(sequence)
        
        # Transpose the sequences to get batches of each component
        
        states = torch.stack([torch.stack([torch.stack(seq[0]).to(device) for seq in sequence]) for sequence in sequences], dim=0)
        actions = torch.stack([torch.stack([torch.tensor(seq[1]).to(device) for seq in sequence]) for sequence in sequences], dim=0)
        rewards = torch.stack([torch.stack([torch.tensor(seq[2]).to(device) for seq in sequence]) for sequence in sequences], dim=0)
        next_states = torch.stack([torch.stack([torch.stack(seq[3]).to(device) for seq in sequence]) for sequence in sequences], dim=0)
        terminations = torch.stack([torch.stack([torch.tensor(seq[4], dtype=torch.int).to(device) for seq in sequence]) for sequence in sequences], dim=2)
        hidden_states = torch.cat([sequence[-1][5] for sequence in sequences], dim = -2).to(device)
        next_hidden_states = torch.cat([sequence[-1][6] for sequence in sequences], dim = -2).to(device)
        
        return (states, actions, rewards, next_states, terminations, hidden_states, next_hidden_states)

    def size(self):
        return len(self.buffer)

# Adjusted epsilon-greedy action function
def epsilon_greedy_action(agent_q_network, state, hidden, epsilon, legal_actions):
    
    if random.random() < epsilon:
        # Explore: take a random action
        action = random.choice(legal_actions)
    else:
        state = state.unsqueeze(0).to(device)  # Add batch dimension
        q_values, hidden = agent_q_network(state, hidden)
        q_values = q_values.cpu().detach().numpy()
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

    return action, hidden


def update_target_network(agent_q_networks, target_q_networks):
    for target, source in zip(target_q_networks, agent_q_networks):
        target.load_state_dict(source.state_dict())


from collections import deque
import torch
from torch.nn.utils.rnn import pad_sequence

def interpolate_weights(start_weights, end_weights, factor):
    return [(1 - factor) * start + factor * end for start, end in zip(start_weights, end_weights)]

def get_dynamic_weights(episode, n_episodes):
    # Define the initial weights for the beginning, middle, and end
    weights_beginning = [0.8, 0.05, 0.05, 0.05, 0.025, 0.025]
    weights_middle = [0.2, 0.25, 0.25, 0.25, 0.025, 0.025]
    weights_end = [0.02, 0.06, 0.06, 0.06, 0.4, 0.4]

    # Calculate the transition points
    transition1 = n_episodes // 3
    transition2 = 2 * n_episodes // 3

    # Adjust weights based on the current episode
    if episode < transition1:
        factor = episode / transition1
        return interpolate_weights(weights_beginning, weights_middle, factor)
    elif episode < transition2:
        factor = (episode - transition1) / (transition2 - transition1)
        return interpolate_weights(weights_middle, weights_end, factor)
    else:
        factor = (episode - transition2) / (n_episodes - transition2)
        return interpolate_weights(weights_middle, weights_end, factor)
    

import os

def create_folder_if_not_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)





        
def prepare_input_sequence(state_buffer, state, sequence_length):
        # Add the current state to the buffer
    state_buffer.append(state)
        
        # If the buffer is not full, pad it with zeros
    if len(state_buffer) < sequence_length:
        padding = [torch.zeros_like(state) for _ in range(sequence_length - len(state_buffer))]
        sequence = padding + list(state_buffer)
    else:
        sequence = list(state_buffer)
        
        # Stack the sequence to form a tensor
    sequence_tensor = torch.stack(sequence)
    
    
    
    
    return sequence_tensor


# Define the environment
def train_qmix(env, name_experiment, agent_q_networks, mixing_network, target_q_networks, replay_buffer, train_p, learn_p, random_enemy=False):
    n_episodes = train_p.n_episodes
    batch_size = train_p.batch_size
    epsilon = train_p.epsilon
    epsilon_min = train_p.epsilon_min
    epsilon_decay = train_p.epsilon_decay
    gamma = train_p.gamma
    target_update_frequency = train_p.target_update_frequency

    lr_agent = learn_p.lr_agent
    lr_mix = learn_p.lr_mix
    learn_gamma = learn_p.gamma

    optimizer = optim.Adam([param for net in agent_q_networks for param in net.parameters()], lr=lr_agent)
    mixing_optimizer = optim.Adam(mixing_network.parameters(), lr=lr_mix)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learn_gamma)
    scheduler_mix = torch.optim.lr_scheduler.ExponentialLR(mixing_optimizer, gamma=learn_gamma)

    steps_counter = 0
    legal_actions = [0, 1, 2, 3, 4]
    agent_indexes = [1, 3]
    episode_rewards = []
    win_rates = []
    enemieName = env.enemieName

    if random_enemy:
        enemylist = ['randomTeam.py', 'baselineTeam.py', 'heuristicTeam.py', 'approxQTeam.py', 'MCTSTeam.py', 'AstarTeam.py']
        weights = get_dynamic_weights(0, n_episodes)
        enemieName = np.random.choice(enemylist, p=weights)

    create_folder_if_not_exists(name_experiment)
    for episode in range(n_episodes):
        print(f"enemy: {enemieName}")
        done = {agent_id: False for agent_id in agent_indexes}
        weights = get_dynamic_weights(episode, n_episodes)
        if random_enemy:
            enemieName = np.random.choice(enemylist, p=weights)
        env.reset(enemieName=enemieName)
        blue_player1_reward = 0
        blue_player2_reward = 0
        score = 0

        hidden_states = torch.stack([torch.stack([agent.init_hidden(1)[0].to(device), agent.init_hidden(1)[1].to(device)])  for agent in agent_q_networks]).clone()
        next_hidden_states = torch.stack([torch.stack([agent.init_hidden(1)[0].to(device), agent.init_hidden(1)[1].to(device)])  for agent in agent_q_networks]).clone()
        
        state_buffers = [deque(maxlen=5) for _ in range(len(agent_indexes))]

        while not all(done.values()):
            actions = [-1 for _, _ in enumerate(env.agents)]
            states = []

            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                state = torch.tensor(obs_agent, dtype=torch.float32).to(device)
                states.append(state)

                
                input_sequence = prepare_input_sequence(state_buffers[i],state, 5)
        
                
                action, next_hidden_states[i, :, :, :, :] = epsilon_greedy_action(agent_q_networks[i], input_sequence, hidden_states[i, :, :, :, :], epsilon, legal_actions)
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
                next_states_converted.append(torch.tensor(next_states[index], dtype=torch.float32))
                rewards_converted.append(rewards[index])
                terminations_converted.append(terminations[index])
                actions_converted.append(actions[index])

            # Ensure hidden states are properly detached before adding to the replay buffer
            replay_buffer.add((states, actions_converted, rewards_converted, next_states_converted, terminations_converted, hidden_states, next_hidden_states))
            
            hidden_states = next_hidden_states.detach().clone()

            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_td_loss(agent_q_networks, mixing_network, target_q_networks, batch, gamma=gamma)
                wandb.log({"loss": loss})

                optimizer.zero_grad()
                mixing_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                mixing_optimizer.step()
                scheduler.step()
                scheduler_mix.step()

        steps_counter += 1
        wandb.log({"epsilon": epsilon})
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        wandb.log({"blue_player1_reward": blue_player1_reward})
        wandb.log({"blue_player2_reward": blue_player2_reward})
        wandb.log({"episode": episode})
        wandb.log({"score": score})
        wandb.log({"enemy": enemieName})

        if (episode + 1) % target_update_frequency == 0:
            update_target_network(agent_q_networks, target_q_networks)

        if (episode + 1) % target_update_frequency == 0:
            torch.save(agent_q_networks[0].state_dict(), f'{name_experiment}/agent1_{episode + 1}.pth')
            torch.save(agent_q_networks[1].state_dict(), f'{name_experiment}/agent3_{episode + 1}.pth')
            torch.save(mixing_network.state_dict(), f'{name_experiment}/mix_{episode + 1}.pth')

        episode_rewards.append(blue_player1_reward + blue_player2_reward)
        print(f"Episode reward: {blue_player1_reward + blue_player2_reward}")

        if np.sum(env.game.state.getBlueFood().data) == 0:
            win_rates.append(1)
        else:
            win_rates.append(0)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_win_rate = np.mean(win_rates[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, Average Reward: {avg_reward:.2f}, Win Rate: {avg_win_rate:.2f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Average Reward per Episode")
    plt.legend()

    plt.subplot(1, 2, 2)
    smoothed_win_rates = np.convolve(win_rates, np.ones(10)/10, mode='valid') if len(win_rates) > 10 else win_rates
    plt.plot(smoothed_win_rates, label="Win Rate (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training completed.")


            


from dataclasses import dataclass
@dataclass
class train_param:
    n_episodes: int = 1
    batch_size: int = 32
    gamma: float = 0.995
    epsilon: float = 0.95
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    target_update_frequency: int = 25
    sequence: bool = False

@dataclass
class learn_param:
    lr_mix: float = 0.001
    lr_agent: float = 0.005
    gamma: float = 0.99

@dataclass
class net_param:
    hidden_dim: int = 64
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2

train_p = train_param()
learn_p = learn_param()
net_p = net_param()

# Get number of agents and their observation/action space dimensions
n_agents = int(len(env.agents) / 2)
action_dim_individual_agent = 5  # North, South, East, West, Stop

obs_individual_agent = env.get_Observation(0)
obs_shape = obs_individual_agent.shape

# parameters


agent_q_networks = [CNNLSTMQNetwork_hidden(embed_dim=64, obs_shape=obs_shape, action_space=action_dim_individual_agent).to(device) for _ in
                    range(n_agents)]

target_q_networks = [CNNLSTMQNetwork_hidden(embed_dim=64, obs_shape=obs_shape, action_space=action_dim_individual_agent).to(device) for _ in
                     range(n_agents)]

# Initialize the mixing network
state_shape = (2, 8, env.layout.height, env.layout.width)  # Veronderstelling: de toestand is een concatenatie van agent-observaties
global_state_shape = [len(env.agents)] + list(obs_shape)
print(global_state_shape)
mixing_network = AttentionMixer(n_agents=n_agents, state_shape=state_shape).to(device)
#mixing_network.load_state_dict(torch.load('mix_75.pth'))

# Initialize target Q-networks with the same weights as the main Q-networks
update_target_network(agent_q_networks, target_q_networks)

# Initialize the replay buffer


import cProfile
import pstats

replay_buffer = ReplayBuffer_sequence()
name_experiment ='LSTM_hidden_tournament2_tiny'
wandb.init(project="", id = f"{name_experiment}__{datetime.now().strftime('%Y%m%d_%H%M%S')}")

env = gymPacMan_parallel_env(display=True, layout_file="layouts/win.lay", length=100, enemieName='randomTeam.py', defenceReward=True)


# Profile the train_qmix function and save the results to a file
profile_filename = 'train_qmix_profile.prof'
train_qmix(env, name_experiment, agent_q_networks, mixing_network, target_q_networks, replay_buffer, train_p, learn_p, random_enemy=False)

# Load the profiling results
profile_stats = pstats.Stats(profile_filename)

# Sort the results by cumulative time and print them
profile_stats.sort_stats('cumtime').print_stats()
wandb.finish()

