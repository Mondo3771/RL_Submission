import crafter
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import os
import json
import numpy as np
from collections import defaultdict


class CrafterMetricsCallback(EvalCallback):
    def __init__(self, *args, metrics_log_path="./metrics_baseline.json", **kwargs):
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


env = crafter.Env()
env = GymV21CompatibilityV0(env=env)

eval_env = crafter.Env()
eval_env = GymV21CompatibilityV0(env=eval_env)
eval_env = Monitor(eval_env)


import os

model = DQN(
    "CnnPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-5,
    buffer_size=1000000,
    learning_starts=50000,
    batch_size=256,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=4,
    target_update_interval=10000,
    exploration_fraction=0.6,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    replay_buffer_kwargs=dict(handle_timeout_termination=False),
    optimize_memory_usage=True,
    tensorboard_log="./dqn_crafter_tensorboard/",
)


checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="dqn_crafter"
)

eval_callback = CrafterMetricsCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=10, 
    metrics_log_path="./metrics_baseline.json"
)

target_timesteps = 2000000 
current_timesteps = model.num_timesteps 
remaining_timesteps = target_timesteps - current_timesteps

if remaining_timesteps > 0:
    model.learn(
        total_timesteps=remaining_timesteps, 
        log_interval=50,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True  # Always reset counter for fresh run
    )
    print(f"\nTraining complete! Total timesteps: {model.num_timesteps:,}")
else:
    print(f"Model already trained to {current_timesteps:,} steps (target: {target_timesteps:,})")
    print("No additional training needed!")

# Save the trained model
model.save("dqn_crafter")
print("Model saved as dqn_crafter")

# Load and test the model
model = DQN.load("dqn_crafter")
print("Testing trained model...")

test_env = crafter.Env()
test_env = crafter.Recorder(
    test_env, './crafter_logs',
    save_stats=True,
    save_video=False,
    save_episode=False,
)
test_env = GymV21CompatibilityV0(env=test_env)

obs, info = test_env.reset()
episode_reward = 0
episode_length = 0

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    episode_reward += reward
    episode_length += 1
    
    if done:
        print(f"Episode finished: Reward={episode_reward}, Length={episode_length}")
        obs, info = test_env.reset()
        episode_reward = 0
        episode_length = 0

test_env.close()


