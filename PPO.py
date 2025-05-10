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
LR = 3e-4
GAMMA = 0.99      # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
CLIP_RATIO = 0.2   # PPO clip ratio
HIDDEN_SIZE = 256  # Size of hidden layers
LSTM_HIDDEN_SIZE = 128  # Size of LSTM hidden state
PPO_EPOCHS = 10    # Number of PPO epochs per update
TARGET_KL = 0.01   # Target KL divergence
ENTROPY_COEF = 0.01  # Entropy coefficient
VALUE_COEF = 0.5    # Value loss coefficient
MAX_GRAD_NORM = 0.5  # Maximum gradient norm
REPLAY_BUFFER_SIZE = 2048
FUTURE_WINDOW = 12  # Number of steps to look ahead for long-term regret

# Define the Crypto Trading Environment (same as before)
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
        
        # Add action bonus to encourage meaningful actions
        action_magnitude = abs(action)
        if action_magnitude > self.action_threshold:
            action_bonus = self.action_bonus_scale * action_magnitude
            
            # Only give bonus if the action was profitable
            if np.sign(action) * np.sign(actual_return) > 0:
                actual_return += action_bonus
        
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

# LSTM-based Actor-Critic Network for PPO
class LSTMActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        super(LSTMActorCritic, self).__init__()
        
        # Shared feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Shared LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(lstm_hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable but state-independent std
        
        # Critic (value) head
        self.critic = nn.Linear(lstm_hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
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
        
        # Policy (actor)
        action_mean = self.actor_mean(lstm_out)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Value (critic)
        value = self.critic(lstm_out)
        
        return action_mean, action_std, value, h
    
    def get_action(self, state, h=None, deterministic=False):
        action_mean, action_std, value, new_h = self.forward(state, h)
        
        if deterministic:
            # For evaluation, use the mean action
            action = torch.tanh(action_mean)
        else:
            # For training, sample from the distribution
            normal = Normal(action_mean, action_std)
            x = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x)
            
        return action, value, new_h
    
    def evaluate_actions(self, states, actions, h=None):
        if h is None:
            action_mean, action_std, value, _ = self.forward(states)
        else:
            action_mean, action_std, value, _ = self.forward(states, h)
        
        # Create normal distribution
        normal = Normal(action_mean, action_std)
        
        # Get log probabilities for actions
        # We need to account for the tanh transformation
        x = torch.atanh(torch.clamp(actions, -0.999, 0.999))  # Inverse of tanh
        log_probs = normal.log_prob(x)
        
        # For continuous actions, we need to account for the change of variables when using tanh
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
        
        # If we have a batch of actions, sum over the action dimensions
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(1, keepdim=True)
        else:
            log_probs = log_probs.sum(0, keepdim=True)
            
        # Calculate entropy of the normal distribution
        entropy = normal.entropy().sum(1, keepdim=True)
        
        return log_probs, entropy, value

# PPO Memory Buffer
class PPOMemory:
    def __init__(self, batch_size=BATCH_SIZE):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size
        
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def size(self):
        return len(self.states)
    
    def generate_batches(self):
        batch_start = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches, np.array(self.states), np.array(self.actions), \
               np.array(self.rewards), np.array(self.values), \
               np.array(self.log_probs), np.array(self.dones)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE,
                 action_threshold=0.2, lr=LR, clip_ratio=CLIP_RATIO):
        self.actor_critic = LSTMActorCritic(state_dim, action_dim, hidden_size, lstm_hidden_size).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.memory = PPOMemory()
        self.clip_ratio = clip_ratio
        self.action_threshold = action_threshold
        
        # Hidden states
        self.hidden = None
        
    def select_action(self, state, evaluate=False, min_action_threshold=None):
        state = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            action, value, new_hidden = self.actor_critic.get_action(state, self.hidden, deterministic=evaluate)
            self.hidden = new_hidden  # Update hidden state
            
            # Get the log probability for the action
            if not evaluate:
                log_probs, _, _ = self.actor_critic.evaluate_actions(state, action, self.hidden)
            else:
                log_probs = torch.zeros(1, 1)  # Dummy value for evaluation
            
            action_np = action.cpu().numpy()
            
            # Apply minimum action threshold for evaluation
            if evaluate and min_action_threshold is not None:
                action_magnitude = np.abs(action_np)
                for i in range(len(action_np)):
                    if 0 < action_magnitude[i] < min_action_threshold:
                        action_np[i] = np.sign(action_np[i]) * min_action_threshold
            
        return action_np, value.cpu().numpy(), log_probs.cpu().numpy()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.memory.push(state, action, reward, value, log_prob, done)
        
    def reset_hidden_states(self):
        self.hidden = None
    
    def update(self):
        # Check if we have enough data
        if self.memory.size() < self.memory.batch_size:
            return {
                'policy_loss': 0,
                'value_loss': 0,
                'entropy': 0,
                'approx_kl': 0,
                'clip_fraction': 0,
                'explained_var': 0
            }
        
        batches, states, actions, rewards, old_values, old_log_probs, dones = self.memory.generate_batches()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_values = torch.FloatTensor(old_values).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        
        # Normalize rewards (optional)
        rewards = torch.FloatTensor(rewards).to(device)
        
        # Compute advantage estimates
        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = 0  # There's no next value after the episode ends
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                    next_non_terminal = 1.0 - dones[t]
                else:
                    next_value = old_values[t + 1]
                    next_non_terminal = 1.0 - dones[t]
                
                delta = rewards[t] + GAMMA * next_value * next_non_terminal - old_values[t]
                advantages[t] = delta + GAMMA * GAE_LAMBDA * next_non_terminal * (
                    advantages[t + 1] if t < len(rewards) - 1 else 0
                )
            
            returns = advantages + old_values
        
        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        clip_fractions = []
        
        for _ in range(PPO_EPOCHS):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                batch_old_values = old_values[batch]
                batch_old_log_probs = old_log_probs[batch]
                
                # Get current log probs and entropy
                new_log_probs, entropy, new_values = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Compute the ratio (policy / old policy)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    # Calculate approximate KL divergence for early stopping
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    
                    # Calculate clipping fraction
                    clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                
                # Update totals
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                clip_fractions.append(clip_fraction)
                n_updates += 1
                
                # Early stopping based on KL divergence
                if approx_kl > TARGET_KL:
                    break
        
        # Clear memory
        self.memory.clear()
        
        # Return metrics
        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_kl / n_updates,
            'clip_fraction': np.mean(clip_fractions)
        }
        
        return metrics
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(directory, filename))
        
    def load(self, directory, filename):
        checkpoint = torch.load(os.path.join(directory, filename))
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

# Training function for PPO
def train(env, agent, num_episodes=100, update_interval=2048, eval_interval=10):
    # Lists to store rewards and losses
    episode_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    best_reward = -float('inf')
    
    # Loop through episodes
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
        consecutive_small_actions = 0
        
        while not done:
            step_count += 1
            
            # Select action
            action, value, log_prob = agent.select_action(state)
            
            # Force occasional actions for exploration
            if abs(action[0]) < 0.1:  # Small action
                consecutive_small_actions += 1
                if consecutive_small_actions > 20:  # After 20 consecutive small actions
                    # Take a random action occasionally to explore
                    if random.random() < 0.3:
                        rand_action = random.uniform(-0.8, 0.8)
                        action[0] = rand_action
                        consecutive_small_actions = 0
            else:
                consecutive_small_actions = 0
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store in PPO memory
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(reward=f"{episode_reward:.4f}")
            
            # Update agent if memory is full
            if agent.memory.size() >= update_interval:
                metrics = agent.update()
                policy_losses.append(metrics['policy_loss'])
                value_losses.append(metrics['value_loss'])
                entropies.append(metrics['entropy'])
                
                print(f"\nUpdate: Pol Loss={metrics['policy_loss']:.4f}, Val Loss={metrics['value_loss']:.4f}, Ent={metrics['entropy']:.4f}, KL={metrics['approx_kl']:.4f}")
        
        # Close progress bar
        pbar.close()
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Calculate action statistics for monitoring
        actions_array = np.array(env.actions_history[1:])  # Skip the initial action (0)
        avg_action_magnitude = np.mean(np.abs(actions_array))
        action_counts = np.sum(np.abs(actions_array) > env.action_threshold)
        action_percentage = (action_counts / len(actions_array)) * 100
        
        # Print episode results
        print(f"Episode {episode+1} completed | Reward: {episode_reward:.4f}")
        print(f"Avg Action Magnitude: {avg_action_magnitude:.4f}")
        print(f"Actions above threshold: {action_counts}/{len(actions_array)} ({action_percentage:.2f}%)")
        
        # Update if any data left in memory
        if agent.memory.size() > 0:
            metrics = agent.update()
            policy_losses.append(metrics['policy_loss'])
            value_losses.append(metrics['value_loss'])
            entropies.append(metrics['entropy'])
       
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            print(f"\nEvaluating after episode {episode+1}...")
            eval_reward = evaluate(env, agent, num_episodes=3)
            print(f"Evaluation reward: {eval_reward:.4f}")
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save("models", "ppo_crypto_best.pt")
                print(f"New best model saved with reward: {best_reward:.4f}")
                
    # Save final model
    agent.save("models", "ppo_crypto_final.pt")
                
    return episode_rewards, policy_losses, value_losses, entropies


# Evaluation function for PPO
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
       non_action_counter = 0  # Count consecutive non-actions
       
       while not done:
           # Get deterministic action from the policy
           action, _, _ = agent.select_action(state, evaluate=True, min_action_threshold=env.action_threshold/2)
           
           # Force actions if agent is being too passive
           if abs(action[0]) < env.action_threshold:
               non_action_counter += 1
               if non_action_counter >= 10:  # If 10+ consecutive small actions
                   # Force an action based on market momentum
                   if env.current_step >= 5:
                       recent_prices = env.df.iloc[env.current_step-5:env.current_step]['close'].values
                       if len(recent_prices) > 1:
                           momentum = (recent_prices[-1] / recent_prices[0]) - 1
                           # Take action based on momentum
                           if abs(momentum) > 0.001:  # If market is moving
                               action[0] = np.sign(momentum) * 0.5  # Take moderately sized action
                               non_action_counter = 0
           else:
               non_action_counter = 0  # Reset counter when action is taken
           
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
   
   print(f"Evaluation average reward: {avg_reward:.4f}")
   print(f"Average action magnitude: {avg_action_magnitude:.4f}")
   print(f"Actions above threshold: {actions_above_threshold}/{len(flat_actions)} ({action_percentage:.2f}%)")
   
   return avg_reward

# Calculate performance metrics (same as before)
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

# Visualization functions (same as before)
def plot_results(episode_rewards, policy_losses, value_losses, entropies):
   """Plot training results."""
   plt.figure(figsize=(15, 10))
   
   # Plot episode rewards
   plt.subplot(2, 2, 1)
   plt.plot(episode_rewards)
   plt.title('Episode Rewards')
   plt.xlabel('Episode')
   plt.ylabel('Reward')
   
   # Plot policy losses
   plt.subplot(2, 2, 2)
   plt.plot(policy_losses)
   plt.title('Policy Losses')
   plt.xlabel('Update Step')
   plt.ylabel('Loss')
   
   # Plot value losses
   plt.subplot(2, 2, 3)
   plt.plot(value_losses)
   plt.title('Value Losses')
   plt.xlabel('Update Step')
   plt.ylabel('Loss')
   
   # Plot entropy
   plt.subplot(2, 2, 4)
   plt.plot(entropies)
   plt.title('Entropy')
   plt.xlabel('Update Step')
   plt.ylabel('Entropy')
   
   plt.tight_layout()
   plt.savefig('training_results_ppo.png')
   plt.show()

def plot_portfolio_performance(portfolio_values, benchmark=None, actions=None):
   """Plot portfolio performance over time with trading actions."""
   fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
   
   # Convert to percentage returns relative to starting value
   initial_value = portfolio_values[0]
   portfolio_returns = [(val / initial_value - 1) * 100 for val in portfolio_values]
   
   # Plot portfolio performance
   axes[0].plot(portfolio_returns, label='PPO Agent')
   
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
   plt.savefig('portfolio_performance_with_actions_ppo.png')
   plt.show()