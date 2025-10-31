import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import crafter
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import json
from collections import defaultdict


class CrafterMetricsCallback(EvalCallback):
    def __init__(self, *args, metrics_log_path="./metrics_dueling.json", **kwargs):
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


class StabilizedDuelingCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256): 
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


class StabilizedDuelingQNetwork(nn.Module):
    def __init__(self, features_extractor, features_dim, action_space):
        super().__init__()
        self.features_extractor = features_extractor
        n_actions = action_space.n
        
        hidden_dim = 128 
        
        self.value_stream = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations):
        features = self.features_extractor(observations)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # This is the "max" variant of Dueling DQN
        q_values = value + (advantages - advantages.max(dim=1, keepdim=True)[0])
        
        return q_values



env = crafter.Env()
env = GymV21CompatibilityV0(env=env)

eval_env = crafter.Env()
eval_env = GymV21CompatibilityV0(env=eval_env)
eval_env = Monitor(eval_env)


policy_kwargs = dict(
    features_extractor_class=StabilizedDuelingCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[], 
    normalize_images=True,
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs=dict(
        eps=1.5e-4, 
    ),
)


model = DQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-5,  
    buffer_size=500000,
    learning_starts=50000,
    batch_size=32,
    tau=0.01,
    gamma=0.99,
    train_freq=8,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10.0,
    replay_buffer_kwargs=dict(handle_timeout_termination=False),
    optimize_memory_usage=True,
    tensorboard_log="./stable_dueling_dqn_tensorboard/",
    verbose=1,
)


checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models_stable_dueling/",
    name_prefix="stable_dueling_dqn"
)

eval_callback = CrafterMetricsCallback(
    eval_env,
    best_model_save_path="./best_model_stable_dueling/",
    log_path="./eval_logs_stable_dueling/",
    eval_freq=10000,
    deterministic=True,
    n_eval_episodes=10,
    metrics_log_path="./metrics_dueling.json"
)


# Train
target_timesteps = 2000000  # Train to 1M total
current_timesteps = model.num_timesteps
remaining_timesteps = target_timesteps - current_timesteps

if remaining_timesteps > 0:
    model.learn(
        total_timesteps=remaining_timesteps,
        log_interval=50,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True
    )
else:
    print("Model already trained to target timesteps!")

model.save("stable_dueling_dqn_crafter")
print("\n✅ Stable Dueling DQN training complete!")
