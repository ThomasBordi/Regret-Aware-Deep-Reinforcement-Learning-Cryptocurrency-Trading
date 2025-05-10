import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gym
from gym import spaces
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
WINDOW_SIZE = 45  # Number of minutes to look back
STEP_SIZE = 5     # Take action every 5 minutes
INITIAL_CAPITAL = 10000  # Initial capital in USDT
TRADING_FEE = 0.001  # 0.1% trading fee
BATCH_SIZE = 64
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.99      # Discount factor
TAU = 0.005       # Target network update rate
HIDDEN_SIZE = 256  # Size of hidden layers
LSTM_HIDDEN_SIZE = 128  # Size of LSTM hidden state
REPLAY_BUFFER_SIZE = 100000
ALPHA = 0.5       # Entropy coefficient - increased for more exploration
TRAINING_EPISODES = 100
FUTURE_WINDOW = 12  # Number of steps to look ahead for long-term regret

# Define the Crypto Trading Environment
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, window_size=WINDOW_SIZE, step_size=STEP_SIZE, initial_capital=INITIAL_CAPITAL, 
             fee=TRADING_FEE, lambda_regret=0.5, use_regret_reward=True, 
             future_window=FUTURE_WINDOW, action_threshold=0.2, action_bonus_scale=0.0005):
        super(CryptoTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.fee = fee
        self.lambda_regret = lambda_regret  # Scaling factor for regret penalty
        self.use_regret_reward = use_regret_reward
        self.future_window = future_window  # How many steps to look ahead for long-term regret
        self.action_threshold = action_threshold  # Threshold for significant actions
        self.action_bonus_scale = action_bonus_scale  # Scale for action bonus reward
        
        # Define action and observation spaces
        # Action space: continuous value between -1 and 1
        # -1 means sell all BTC, 1 means use all USDT to buy BTC, 0 means hold
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: OHLCV data + indicators + portfolio state
        feature_count = len(df.columns)
        # window_size time steps + 2 additional features for portfolio state (cash ratio and crypto ratio)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size * feature_count + 2,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()

    def reset(self):
        # Reset to the beginning of data (after window_size)
        self.current_step = self.window_size
        self.cash = self.initial_capital
        self.btc = 0
        self.total_value_history = [self.initial_capital]
        self.actions_history = [0]  # Start with no action
        self.done = False
        
        return self._get_observation()

    def _get_observation(self):
        # Get the window of data
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        
        # Get portfolio state (cash ratio and btc ratio)
        current_price = self.df.iloc[self.current_step-1]['close']
        total_value = self.cash + self.btc * current_price
        cash_ratio = self.cash / total_value if total_value > 0 else 0
        btc_ratio = (self.btc * current_price) / total_value if total_value > 0 else 0
        
        # Flatten window data and add portfolio state
        obs = np.append(window_data.flatten(), [cash_ratio, btc_ratio])
        return obs.astype(np.float32)

    def _calculate_max_possible_value(self, V_prev, current_price, previous_price):
        """Calculate maximum possible portfolio value using perfect foresight"""
        market_return = (current_price - previous_price) / previous_price
        
        if market_return > 0:
            # If market went up, best action would be to be fully invested in BTC
            # Assuming we could convert all cash to BTC at previous step without fees
            return V_prev * (1 + market_return)
        else:
            # If market went down, best action would be to hold cash
            return V_prev
    
    def _calculate_long_term_regret(self, total_value_before, current_step):
        """Calculate long-term regret by comparing holding vs optimal strategy over multiple steps"""
        # Store original portfolio state
        original_cash = self.cash
        original_btc = self.btc
        original_step = self.current_step
        
        # Make sure we don't go beyond available data
        max_future_steps = min(self.future_window, len(self.df) - current_step)
        if max_future_steps <= 1:
            # Fallback to immediate regret if we can't look ahead
            current_price = self.df.iloc[current_step - self.step_size - 1]['close']
            new_price = self.df.iloc[current_step - 1]['close']
            
            # Calculate maximum possible value using perfect foresight
            max_possible_value = self._calculate_max_possible_value(
                total_value_before, new_price, current_price
            )
            
            # Current portfolio value after action
            total_value_after = self.cash + self.btc * new_price
            actual_return = (total_value_after - total_value_before) / total_value_before
            
            # Calculate maximum possible return
            max_possible_return = (max_possible_value - total_value_before) / total_value_before
            
            # Calculate regret (difference between max possible return and actual return)
            return max_possible_return - actual_return
        
        # 1. Calculate value if we just held the current portfolio
        future_values_holding = []
        current_btc = original_btc
        current_cash = original_cash
        
        for i in range(max_future_steps):
            if current_step + i >= len(self.df):
                break
            future_price = self.df.iloc[current_step + i - 1]['close']
            future_value = current_cash + current_btc * future_price
            future_values_holding.append(future_value)
        
        max_holding_value = max(future_values_holding) if future_values_holding else total_value_before
        holding_return = (max_holding_value - total_value_before) / total_value_before
        
        # 2. Calculate optimal strategy with perfect foresight
        # Reset to original state
        self.cash = original_cash
        self.btc = original_btc
        max_perfect_value = total_value_before
        
        for i in range(1, max_future_steps):
            if current_step + i >= len(self.df):
                break
                
            current_price = self.df.iloc[current_step + i - 2]['close']
            next_price = self.df.iloc[current_step + i - 1]['close']
            
            # If price will increase, convert all to BTC
            if next_price > current_price:
                if self.cash > 0:
                    additional_btc = (self.cash / current_price) * (1 - self.fee)
                    self.btc += additional_btc
                    self.cash = 0
            # If price will decrease, convert all to cash
            else:
                if self.btc > 0:
                    additional_cash = (self.btc * current_price) * (1 - self.fee)
                    self.cash += additional_cash
                    self.btc = 0
            
            # Calculate portfolio value at this step
            future_value = self.cash + self.btc * next_price
            max_perfect_value = max(max_perfect_value, future_value)
        
        # Calculate optimal return
        optimal_return = (max_perfect_value - total_value_before) / total_value_before
        
        # Reset portfolio to original state
        self.cash = original_cash
        self.btc = original_btc
        self.current_step = original_step
        
        # Long-term regret is difference between optimal and holding returns
        return optimal_return - holding_return
    
    def step(self, action):
        # Ensure we haven't reached the end of data
        if self.current_step >= len(self.df) - self.step_size:
            self.done = True
            return self._get_observation(), 0, self.done, {}
        
        # Get current price and total value before action
        current_price = self.df.iloc[self.current_step-1]['close']
        total_value_before = self.cash + self.btc * current_price
        
        # Store previous action for comparison
        prev_action = self.actions_history[-1]
        
        # Extract action (ensure it's a scalar)
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)  # Clip action to ensure it's within bounds
        
        # Store the current action
        self.actions_history.append(action)
        
        # Execute trade based on action
        if action > 0:  # Buy BTC
            # Calculate amount to buy
            btc_to_buy = (self.cash * action) / current_price
            # Apply trading fee
            btc_to_buy *= (1 - self.fee)
            # Update portfolio
            self.btc += btc_to_buy
            self.cash -= btc_to_buy * current_price
        elif action < 0:  # Sell BTC
            # Calculate amount to sell
            btc_to_sell = self.btc * abs(action)
            # Apply trading fee
            usdt_received = btc_to_sell * current_price * (1 - self.fee)
            # Update portfolio
            self.btc -= btc_to_sell
            self.cash += usdt_received
        
        # Move forward in time by step_size
        self.current_step += self.step_size
        
        # Calculate reward (simple return from before to after action)
        new_price = self.df.iloc[self.current_step-1]['close']
        total_value_after = self.cash + self.btc * new_price
        self.total_value_history.append(total_value_after)
        
        # Basic reward (percentage change in portfolio value)
        actual_return = (total_value_after - total_value_before) / total_value_before
        
        if self.use_regret_reward:
            # Calculate long-term regret
            regret = self._calculate_long_term_regret(total_value_before, self.current_step)
            
            # Calculate market volatility over recent window
            volatility_window = min(20, self.current_step)  # Use up to 20 steps
            price_history = []
            for i in range(volatility_window):
                idx = self.current_step - i - 1
                if idx >= 0:
                    price_history.append(self.df.iloc[idx]['close'])
            
            price_history = np.array(price_history)
            if len(price_history) > 1:
                returns = np.diff(price_history) / price_history[:-1]
                market_volatility = np.std(returns) + 1e-6  # Add small constant to avoid division by zero
            else:
                market_volatility = 0.01  # Default if not enough history
            
            # Normalize both components by market volatility
            normalized_return = actual_return / market_volatility
            normalized_regret = regret / market_volatility
            
            # Final reward with normalized components
            reward = normalized_return - self.lambda_regret * normalized_regret
        else:
            # Use traditional reward if regret-based reward is disabled
            reward = actual_return
        
        # Check if we're done
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        # Implement visualization if needed
        pass

# LSTM-based Actor Network
class LSTMActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        super(LSTMActorNetwork, self).__init__()
        
        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(lstm_hidden_size, action_dim)
        self.log_std_layer = nn.Linear(lstm_hidden_size, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
                
    def forward(self, state, h=None):
        # Ensure state has the right shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Extract features
        x = self.feature_layer(state)
        
        # Reshape for LSTM: [batch_size, sequence_length=1, hidden_size]
        x = x.unsqueeze(1)
        
        # Process through LSTM
        if h is None:
            lstm_out, h = self.lstm(x)
        else:
            lstm_out, h = self.lstm(x, h)
            
        lstm_out = lstm_out.squeeze(1)
        
        # Output mean and log_std
        mean = self.mean_layer(lstm_out)
        log_std = self.log_std_layer(lstm_out)
        
        # Constrain log_std to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std, h
    
    def sample(self, state, h=None):
        mean, log_std, h = self.forward(state, h)
        std = log_std.exp()
        
        # Sample from Gaussian distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Calculate log probability for training
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        
        # If we have a batch of actions, sum over the action dimensions
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum(0, keepdim=True)
            
        return action, log_prob, h

# LSTM-based Critic Network
class LSTMCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        super(LSTMCriticNetwork, self).__init__()
        
        # Q1 Network
        self.q1_feature_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.q1_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.q1_output = nn.Linear(lstm_hidden_size, 1)
        
        # Q2 Network (for reducing overestimation bias)
        self.q2_feature_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.q2_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.q2_output = nn.Linear(lstm_hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
                
    def forward(self, state, action, h1=None, h2=None):
        # Ensure state has the right shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Ensure action has the right shape and dimensions
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(state.device)
            
        if len(action.shape) == 1:
            # If action is a 1D tensor, reshape it to match state batch size
            if action.size(0) == 1:
                # If it's a single action, expand it to match state batch
                action = action.repeat(state.size(0))
            action = action.unsqueeze(-1)  # Add feature dimension
            
        elif len(action.shape) == 2 and action.size(1) == 1:
            # If action is [batch_size, 1], keep it as is
            pass
        elif len(action.shape) == 2 and action.size(0) == 1:
            # If action is [1, features], expand to match state batch
            action = action.repeat(state.size(0), 1)
        elif len(action.shape) == 3:
            # If action is a 3D tensor [batch, seq_len, features], squeeze the middle dim
            action = action.squeeze(1)
            
        # Concatenate state and action
        try:
            x = torch.cat([state, action], dim=1)
        except RuntimeError as e:
            print(f"State shape: {state.shape}, Action shape: {action.shape}")
            raise e
        
        # Process through Q1 network
        q1_features = self.q1_feature_layer(x)
        q1_features = q1_features.unsqueeze(1)  # Add sequence dimension
        
        if h1 is None:
            q1_lstm_out, h1 = self.q1_lstm(q1_features)
        else:
            q1_lstm_out, h1 = self.q1_lstm(q1_features, h1)
            
        q1_lstm_out = q1_lstm_out.squeeze(1)
        q1 = self.q1_output(q1_lstm_out)
        
        # Process through Q2 network
        q2_features = self.q2_feature_layer(x)
        q2_features = q2_features.unsqueeze(1)  # Add sequence dimension
        
        if h2 is None:
            q2_lstm_out, h2 = self.q2_lstm(q2_features)
        else:
            q2_lstm_out, h2 = self.q2_lstm(q2_features, h2)
            
        q2_lstm_out = q2_lstm_out.squeeze(1)
        q2 = self.q2_output(q2_lstm_out)
        
        return q1, q2, h1, h2

# Experience Replay Buffer with prioritization
class PrioritizedReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE, alpha=0.6, beta=0.4, action_threshold=0.2):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization factor
        self.beta = beta    # Importance sampling factor
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        self.action_threshold = action_threshold
        
    def push(self, state, action, reward, next_state, done):
        # Store experience with priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Set priority based on action magnitude and reward
        action_magnitude = float(abs(action[0])) if isinstance(action, np.ndarray) else float(abs(action))
        reward_value = float(reward)
        
        # Higher priority for successful significant actions
        if action_magnitude > self.action_threshold and reward_value > 0:
            priority = self.max_priority * 2.0
        # Medium priority for significant actions (even if not successful)
        elif action_magnitude > self.action_threshold:
            priority = self.max_priority * 1.3
        # Normal priority for holding actions
        else:
            priority = self.max_priority
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(probs), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        weights = torch.FloatTensor(weights).to(device)
        
        # Gather experiences
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
        self.max_priority = max(self.max_priority, priorities.max())
        
    def size(self):
        return len(self.buffer)

# Soft Actor-Critic Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE,
                 action_threshold=0.2, initial_alpha=0.5, target_entropy_scale=0.8):
        self.actor = LSTMActorNetwork(state_dim, action_dim, hidden_size, lstm_hidden_size).to(device)
        self.critic = LSTMCriticNetwork(state_dim, action_dim, hidden_size, lstm_hidden_size).to(device)
        self.critic_target = LSTMCriticNetwork(state_dim, action_dim, hidden_size, lstm_hidden_size).to(device)
        
        # Copy weights from critic to critic_target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target network parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False
            
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Initialize replay buffer with action threshold
        self.replay_buffer = PrioritizedReplayBuffer(action_threshold=action_threshold)
        
        # Initialize temperature parameter alpha with higher value to encourage exploration
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device=device)
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ACTOR)
        
        # Set target entropy to -dim(A) * scale (less negative to encourage exploration)
        self.target_entropy = -action_dim * target_entropy_scale
        
        # Store action threshold
        self.action_threshold = action_threshold
        
        # Initialize hidden states
        self.actor_hidden = None
        self.critic_hidden1 = None
        self.critic_hidden2 = None
        
        # Action history for monitoring
        self.action_history = []
        
    def select_action(self, state, evaluate=False, min_action_threshold=None):
        state = torch.FloatTensor(state).to(device)
        
        if evaluate:  # Deterministic action for evaluation
            with torch.no_grad():
                mean, _, self.actor_hidden = self.actor(state, self.actor_hidden)
                action = torch.tanh(mean).cpu().numpy()
                self.action_history.append(action[0])
                return action
        else:  # Stochastic action for training
            with torch.no_grad():
                action, _, self.actor_hidden = self.actor.sample(state, self.actor_hidden)
                action_np = action.cpu().numpy()
                return action_np
    
    def reset_hidden_states(self):
        self.actor_hidden = None
        self.critic_hidden1 = None
        self.critic_hidden2 = None
        self.action_history = []
                
    def update(self, batch_size=BATCH_SIZE):
        # Only update if we have enough samples
        if self.replay_buffer.size() < batch_size:
            return 0, 0, 0
            
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return 0, 0, 0
            
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Extract action magnitudes
        action_magnitudes = torch.abs(actions).mean().item()
        
        # Dynamically adjust target entropy based on agent's behavior
        if action_magnitudes < self.action_threshold:
            # If agent is taking small actions, increase entropy to encourage exploration
            adjusted_target_entropy = self.target_entropy * 1.5
        else:
            # Normal entropy target
            adjusted_target_entropy = self.target_entropy

        # Get current Q values
        q1_values, q2_values, _, _ = self.critic(states, actions)
        
        # Compute target Q values (without gradients)
        with torch.no_grad():
            # Sample actions from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Get Q values from target critic
            next_q1, next_q2, _, _ = self.critic_target(next_states, next_actions)
            
            # Take the minimum of the two Q values to reduce overestimation
            next_q_values = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            
            # Compute target Q value using Bellman equation
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Calculate critic loss (weighted MSE)
        critic_loss1 = F.mse_loss(q1_values, target_q_values, reduction='none')
        critic_loss2 = F.mse_loss(q2_values, target_q_values, reduction='none')
        
        # Apply importance sampling weights
        critic_loss1 = (weights * critic_loss1).mean()
        critic_loss2 = (weights * critic_loss2).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update priorities in replay buffer
        td_errors = torch.abs(q1_values - target_q_values) + torch.abs(q2_values - target_q_values)
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6  # Add small constant to avoid zero priority
        self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Update actor
        # Sample actions and log probabilities from current policy
        new_actions, log_probs, _ = self.actor.sample(states)
        
        # Get Q values from current critic
        q1, q2, _, _ = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)
        
        # Calculate actor loss
        actor_loss = (self.alpha * log_probs - min_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter using adjusted target entropy
        alpha_loss = -(self.log_alpha * (log_probs.detach() + adjusted_target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = torch.exp(self.log_alpha)
        
        # Soft update of target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, os.path.join(directory, filename))
        
    def load(self, directory, filename):
        checkpoint = torch.load(os.path.join(directory, filename))
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

def train(env, agent, num_episodes=TRAINING_EPISODES, batch_size=BATCH_SIZE, eval_interval=10):
    # Lists to store rewards and losses
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    alpha_losses = []
    best_reward = -float('inf')
    
    # Loop through episodes with progress bar
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        state = env.reset()
        agent.reset_hidden_states()
        episode_reward = 0
        done = False
        
        # Estimate total steps for this episode
        max_steps = (len(env.df) - env.window_size) // env.step_size
        
        # Create progress bar for steps
        pbar = tqdm(total=max_steps, desc="Training", miniters=1)
        
        step_count = 0
        
        while not done:
            step_count += 1
            
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(reward=f"{episode_reward:.4f}")
            
            # Update agent
            if agent.replay_buffer.size() > batch_size:
                critic_loss, actor_loss, alpha_loss = agent.update(batch_size)
                
                if critic_loss != 0:  # Only append non-zero losses
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                    alpha_losses.append(alpha_loss)
        
        # Close progress bar
        pbar.close()
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Calculate average action magnitude to track agent's behavior
        actions_array = np.array(env.actions_history[1:])  # Skip the initial action (0)
        avg_action_magnitude = np.mean(np.abs(actions_array))
        action_counts = np.sum(np.abs(actions_array) > env.action_threshold)
        action_percentage = (action_counts / len(actions_array)) * 100
        
        # Print episode results
        print(f"Episode {episode+1} completed | Reward: {episode_reward:.4f}")
        print(f"Alpha: {agent.alpha.item():.4f} | Avg Action Magnitude: {avg_action_magnitude:.4f}")
        print(f"Actions above threshold: {action_counts}/{len(actions_array)} ({action_percentage:.2f}%)")
        
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            print(f"\nEvaluating after episode {episode+1}...")
            eval_reward = evaluate(env, agent, num_episodes=3)
            print(f"Evaluation reward: {eval_reward:.4f}")
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save("models", "sac_crypto_best.pt")
                print(f"New best model saved with reward: {best_reward:.4f}")
                
    # Save final model
    agent.save("models", "sac_crypto_final.pt")
                
    return episode_rewards, critic_losses, actor_losses, alpha_losses

# Evaluation function
def evaluate(env, agent, num_episodes=5, render=False):
    rewards = []
    actions_data = []
    
    # Create progress bar for evaluation episodes
    eval_bar = tqdm(range(num_episodes), desc="Evaluating")
    
    for episode in eval_bar:
        state = env.reset()
        agent.reset_hidden_states()
        episode_reward = 0
        done = False
        
        # Track progress within episode
        steps = 0
        episode_actions = []
        
        while not done:
            # Select action deterministically
            action = agent.select_action(state, evaluate=True)
            episode_actions.append(action[0])
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Update progress bar
            eval_bar.set_postfix(reward=f"{episode_reward:.2f}", steps=steps)
            
            if render:
                env.render()
                
        rewards.append(episode_reward)
        actions_data.append(episode_actions)
        
    # Calculate average reward
    avg_reward = np.mean(rewards)
    
    # Print action statistics
    flat_actions = [a for episode_actions in actions_data for a in episode_actions]
    avg_action_magnitude = np.mean(np.abs(flat_actions))
    actions_above_threshold = sum(1 for a in flat_actions if abs(a) >= env.action_threshold)
    action_percentage = (actions_above_threshold / len(flat_actions)) * 100
    
    print(f"Evaluation average action magnitude: {avg_action_magnitude:.4f}")
    print(f"Actions above threshold: {actions_above_threshold}/{len(flat_actions)} ({action_percentage:.2f}%)")
    
    return avg_reward

# Calculate performance metrics
def calculate_metrics(portfolio_values):
    """Calculate key trading performance metrics."""
    portfolio_values = np.array(portfolio_values)
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Total Return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Annualized Return (assuming 252 trading days per year, 24*60 minutes per day)
    n_minutes = len(portfolio_values) - 1
    annualized_return = (1 + total_return) ** (252 * 24 * 60 / n_minutes) - 1
    
    # Volatility (annualized)
    daily_vol = np.std(returns) * np.sqrt(24 * 60)  # Daily volatility (24*60 minutes)
    annualized_vol = daily_vol * np.sqrt(252)  # Annualized volatility
    
    # Sharpe Ratio (assuming risk-free rate of 0)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Sortino Ratio (penalizes only negative volatility)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(24 * 60 * 252) if len(negative_returns) > 0 else 0  # Annualized
    sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
    
    # Calmar Ratio (return / max drawdown)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
    
    # Win Rate (percentage of profitable trades)
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate
    }
    
    return metrics

# Visualization functions
def plot_results(episode_rewards, critic_losses, actor_losses, alpha_losses):
    """Plot training results."""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot critic losses
    plt.subplot(2, 2, 2)
    plt.plot(critic_losses)
    plt.title('Critic Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    
    # Plot actor losses
    plt.subplot(2, 2, 3)
    plt.plot(actor_losses)
    plt.title('Actor Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    
    # Plot alpha losses
    plt.subplot(2, 2, 4)
    plt.plot(alpha_losses)
    plt.title('Alpha Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def plot_portfolio_performance(portfolio_values, benchmark=None, actions=None):
    """Plot portfolio performance over time with trading actions."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Convert to percentage returns relative to starting value
    initial_value = portfolio_values[0]
    portfolio_returns = [(val / initial_value - 1) * 100 for val in portfolio_values]
    
    # Plot portfolio performance
    axes[0].plot(portfolio_returns, label='SAC Agent')
    
    # Plot benchmark if provided
    if benchmark is not None:
        initial_benchmark = benchmark[0]
        benchmark_returns = [(val / initial_benchmark - 1) * 100 for val in benchmark]
        axes[0].plot(benchmark_returns, label='Buy and Hold', color='green', linestyle='--')
    
    axes[0].set_title('Portfolio Performance')
    axes[0].set_ylabel('Return (%)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot actions if provided
    if actions is not None:
        # Skip the first action (0) in display
        axes[1].plot(actions[1:], color='red')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        # Add thresholds
        axes[1].axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-0.2, color='green', linestyle='--', alpha=0.5)
        axes[1].set_title('Trading Actions')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Action (-1 to 1)')
        axes[1].set_ylim(-1.1, 1.1)
        
    plt.tight_layout()
    plt.savefig('portfolio_performance_with_actions.png')
    plt.show()