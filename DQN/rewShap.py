import crafter
import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import json
import numpy as np
from collections import defaultdict


class CrafterMetricsCallback(EvalCallback):
    def __init__(self, *args, metrics_log_path="./metrics_shaped.json", **kwargs):
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
            # Track achievements and metrics
            achievement_episode_counts = defaultdict(int)
            survival_times = []
            cumulative_rewards = []

            # Run evaluation episodes
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
                    
                    # Store achievements from each step
                    if 'achievements' in info:
                        final_achievements = info['achievements']

                if final_achievements is not None:
                    for achievement, count in final_achievements.items():
                        if count > 0:
                            achievement_episode_counts[achievement] += 1

                survival_times.append(episode_length)
                cumulative_rewards.append(episode_reward)

            # Calculate achievement unlock rates (percentage across episodes)
            achievement_rates = {k: float((v / self.n_eval_episodes) * 100)
                                for k, v in achievement_episode_counts.items()}

            # Calculate geometric mean (Crafter's official metric)
            if achievement_rates:
                rates_values = list(achievement_rates.values())
                geometric_mean = float(np.exp(np.mean(np.log(np.array(rates_values) + 1e-8))))
            else:
                geometric_mean = 0.0

            # Store metrics (convert numpy types to Python native types)
            self.all_metrics['timesteps'].append(int(self.num_timesteps))
            self.all_metrics['achievement_rates'].append(achievement_rates)
            self.all_metrics['geometric_means'].append(geometric_mean)
            self.all_metrics['survival_times'].append(float(np.mean(survival_times)))
            self.all_metrics['cumulative_rewards'].append(float(np.mean(cumulative_rewards)))
            self.all_metrics['eval_rewards'].append(float(np.mean(self.evaluations_results[-1])))

            # Save metrics to file
            with open(self.metrics_log_path, 'w') as f:
                json.dump(self.all_metrics, f, indent=2)
        
        return result


class BalancedRewardShaping(gym.Wrapper):
    def __init__(self, env, shaping_weight=0.05):
        super().__init__(env)
        self.shaping_weight = shaping_weight 
        self.visited_positions = set()
        self.last_health = 9
        self.last_hunger = 9
        self.last_achievements = 0
        
    def reset(self, **kwargs):
        self.visited_positions = set()
        self.last_health = 9
        self.last_hunger = 9
        self.last_achievements = 0
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        original_reward = reward
        shaped_bonus = 0.0
        
        # Tiny auxiliary rewards
        if hasattr(self.env.unwrapped, 'player_pos'):
            pos = tuple(self.env.unwrapped.player_pos)
            if pos not in self.visited_positions:
                shaped_bonus += 0.001
                self.visited_positions.add(pos)
        
        if 'health' in info:
            health_delta = info['health'] - self.last_health
            shaped_bonus += 0.01 if health_delta > 0 else -0.005
            self.last_health = info['health']
        
        if 'hunger' in info:
            if info['hunger'] > self.last_hunger:
                shaped_bonus += 0.01
            self.last_hunger = info['hunger']
        
        # BIG bonus for achievements (on top of original +1.0)
        if 'achievements' in info:
            achievements = sum(info['achievements'].values())
            if achievements > self.last_achievements:
                shaped_bonus += 10.0 
                self.last_achievements = achievements
        
        # Tiny survival bonus
        shaped_bonus += 0.0001

        if terminated:
            # Small death penalty
            shaped_bonus -= 0.1  
        
        # Final reward: original (dominant) + small shaped bonus
        final_reward = original_reward + (self.shaping_weight * shaped_bonus)
        
        return obs, final_reward, terminated, truncated, info



train_env = crafter.Env()
train_env = GymV21CompatibilityV0(env=train_env)
train_env = BalancedRewardShaping(train_env, shaping_weight=0.05)
train_env = Monitor(train_env, "./train_logs_shaped")


eval_env_true = crafter.Env()
eval_env_true = GymV21CompatibilityV0(env=eval_env_true)
eval_env_true = Monitor(eval_env_true, "./eval_logs_true")


eval_env_shaped = crafter.Env()
eval_env_shaped = GymV21CompatibilityV0(env=eval_env_shaped)
eval_env_shaped = BalancedRewardShaping(eval_env_shaped, shaping_weight=0.05)
eval_env_shaped = Monitor(eval_env_shaped, "./eval_logs_shaped")


policy_kwargs = dict(
    net_arch=[256, 256],
    normalize_images=True,
)


model = DQN(
    "CnnPolicy",
    train_env,
    policy_kwargs=policy_kwargs,
    learning_rate=5e-5,
    buffer_size=1000000,
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
    replay_buffer_kwargs=dict(handle_timeout_termination=False),
    optimize_memory_usage=True,
    tensorboard_log="./fixed_shaped_dqn_tensorboard/",
    verbose=1,
)


checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models_fixed_shaped/",
    name_prefix="fixed_shaped_dqn"
)

# Monitor BOTH true and shaped performance
eval_callback_true = CrafterMetricsCallback(
    eval_env_true,
    best_model_save_path="./best_model_true/",
    log_path="./eval_logs_true/",
    eval_freq=10000,
    deterministic=True,
    n_eval_episodes=10,
    verbose=1,
    metrics_log_path="./metrics_shaped.json"
)

eval_callback_shaped = EvalCallback(
    eval_env_shaped,
    best_model_save_path="./best_model_shaped/",
    log_path="./eval_logs_shaped/",
    eval_freq=10000,
    deterministic=True,
    n_eval_episodes=10,
    verbose=1,
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
        callback=[checkpoint_callback, eval_callback_true, eval_callback_shaped],
        reset_num_timesteps=True
    )
else:
    print("Model already trained to target timesteps!")

model.save("fixed_shaped_dqn_crafter")
print("\n✅ Training complete! Check eval_logs_true/ for TRUE performance")