import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch
import crafter
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import NamedTuple, Optional
import torch.nn.functional as F
import os
import json
from collections import defaultdict


class CrafterMetricsCallback(EvalCallback):
    def __init__(self, *args, metrics_log_path="./metrics_per.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_log_path = metrics_log_path
        self.all_metrics = {
            'timesteps': [],
            'achievement_rates': [],
            'geometric_means': [],
            'survival_times': [],
            'cumulative_rewards': [],
            'eval_rewards': []
        }
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            achievement_episode_counts = defaultdict(int)
            survival_times = []
            cumulative_rewards = []

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                final_achievements = None

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    if isinstance(info, list):
                        info = info[0] if len(info) > 0 else {}
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if 'achievements' in info:
                        final_achievements = info['achievements']

                if final_achievements is not None:
                    for achievement, count in final_achievements.items():
                        if count > 0:
                            achievement_episode_counts[achievement] += 1

                survival_times.append(episode_length)
                cumulative_rewards.append(episode_reward)

            achievement_rates = {k: float((v / self.n_eval_episodes) * 100)
                                for k, v in achievement_episode_counts.items()}

            if achievement_rates:
                rates_values = list(achievement_rates.values())
                geometric_mean = float(np.exp(np.mean(np.log(np.array(rates_values) + 1e-8))))
            else:
                geometric_mean = 0.0

            self.all_metrics['timesteps'].append(int(self.num_timesteps))
            self.all_metrics['achievement_rates'].append(achievement_rates)
            self.all_metrics['geometric_means'].append(geometric_mean)
            self.all_metrics['survival_times'].append(float(np.mean(survival_times)))
            self.all_metrics['cumulative_rewards'].append(float(np.mean(cumulative_rewards)))
            self.all_metrics['eval_rewards'].append(float(np.mean(self.evaluations_results[-1])))

            with open(self.metrics_log_path, 'w') as f:
                json.dump(self.all_metrics, f, indent=2)
        
        return result


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: torch.Tensor 
    indices: np.ndarray 


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self, 
        buffer_size, 
        observation_space, 
        action_space, 
        device,
        alpha=0.6, 
        beta=0.4, 
        beta_increment=0.001, 
        **kwargs
    ):
        super().__init__(buffer_size, observation_space, action_space, device, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0
        self.eps = 1e-6
    
    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx] = self.max_priority
    
    def sample(self, batch_size, env=None):
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probs = (priorities + self.eps) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(probs), batch_size, p=probs, replace=False)
        
        total = len(probs)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        samples = self._get_samples(indices, env)

        return PrioritizedReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            weights=torch.FloatTensor(weights).to(self.device).reshape(-1, 1),
            indices=indices
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(priority) + self.eps
            self.max_priority = max(self.max_priority, abs(priority) + self.eps)


class CustomDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                # Compute target Q values
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute TD errors for priority updates
            td_errors = target_q_values - current_q_values
            
            # Apply importance sampling weights if using PER
            if isinstance(replay_data, PrioritizedReplayBufferSamples):
                # Weighted MSE loss
                loss = (replay_data.weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
                
                # Update priorities in the replay buffer
                priorities = td_errors.detach().cpu().numpy().flatten()
                self.replay_buffer.update_priorities(replay_data.indices, priorities)
            else:
                # Standard MSE loss
                loss = F.mse_loss(current_q_values, target_q_values)

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))



env = crafter.Env()
env = GymV21CompatibilityV0(env=env)

eval_env = crafter.Env()
eval_env = GymV21CompatibilityV0(env=eval_env)
eval_env = Monitor(eval_env)

policy_kwargs = dict(
    net_arch=[256, 256],
    normalize_images=True,
)



# Create DQN with PER - using CustomDQN class
model = CustomDQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=5e-5,  
    buffer_size=500000,
    learning_starts=50000,
    batch_size=64,  
    tau=0.01, 
    gamma=0.99,
    train_freq=4,
    gradient_steps=2,
    target_update_interval=2000,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10.0,
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs=dict(
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        handle_timeout_termination=False
    ),
    optimize_memory_usage=True,
    tensorboard_log="./per_dqn_tensorboard/",
    verbose=1,
)


checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models_per/",
    name_prefix="per_dqn"
)

eval_callback = CrafterMetricsCallback(
    eval_env,
    best_model_save_path="./best_model_per/",
    log_path="./eval_logs_per/",
    eval_freq=10000,
    deterministic=True,
    n_eval_episodes=10,
    metrics_log_path="./metrics_per.json"
)

# Train
target_timesteps = 2000000
current_timesteps = model.num_timesteps
remaining_timesteps = target_timesteps - current_timesteps

print(f"\nTraining from {current_timesteps:,} to {target_timesteps:,} timesteps")
print(f"Remaining: {remaining_timesteps:,} timesteps\n")

if remaining_timesteps > 0:
    model.learn(
        total_timesteps=remaining_timesteps,
        log_interval=50,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True
    )
else:
    print("Model already trained to target timesteps!")

model.save("per_dqn_crafter")
print("\n✅ PER DQN training complete!")