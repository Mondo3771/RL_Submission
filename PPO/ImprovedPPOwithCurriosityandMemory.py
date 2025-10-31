import crafter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import NatureCNN
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import os
import argparse
from typing import Callable, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import defaultdict


# ==============================================================================
# ICM AND ICM_PPO CLASS DEFINITIONS
# ==============================================================================

class ICM(nn.Module):
    """Intrinsic Curiosity Module"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, eta=0.2, beta=0.2, lr=1e-3):
        super(ICM, self).__init__()
        self.eta = eta
        self.beta = beta
        self.action_dim = action_dim
        
        # Calculate feature size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            conv_layers = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            feature_size = conv_layers(dummy_input).shape[1]
        
        print(f"ICM Feature size: {feature_size}") 
        
        self.feature_net = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU()
        )
        
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state, next_state, action):
        # Normalize uint8 to float32 [0, 1]
        state = state.float() / 255.0
        next_state = next_state.float() / 255.0
        
        state_feat = self.feature_net(state)
        next_state_feat = self.feature_net(next_state)

        if action is None:
            raise ValueError("Action tensor is None")
        
        action_indices = action.view(-1).long()
        action_one_hot = F.one_hot(action_indices, num_classes=self.action_dim).float().to(action.device)

        # Inverse model loss
        pred_action_logits = self.inverse_net(torch.cat([state_feat, next_state_feat], dim=1))
        inverse_loss = F.cross_entropy(pred_action_logits, action_indices)
        
        # Forward model loss
        forward_input = torch.cat([state_feat, action_one_hot], dim=1)
        pred_next_state_feat = self.forward_net(forward_input)

        per_element_loss = F.mse_loss(pred_next_state_feat, next_state_feat, reduction='none')
        forward_loss_per_sample = per_element_loss.mean(dim=1)
        forward_loss = forward_loss_per_sample.mean()

        # Intrinsic reward
        intrinsic_reward = self.eta * forward_loss_per_sample.detach()

        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        return intrinsic_reward, total_loss
    
class ICM_PPO(PPO):
    """PPO with Intrinsic Curiosity Module"""
    def __init__(self, *args, **kwargs):
        self.icm = None
        self.icm_kwargs = kwargs.pop('icm_kwargs', {})
        self.n_stack = kwargs.pop('frame_stack', 4)
        self.intrinsic_reward_scale = kwargs.pop('intrinsic_reward_scale', 0.01)
        super(ICM_PPO, self).__init__(*args, **kwargs)
        
    def _init_icm(self, observation_space, action_space):
        """Initialize ICM module"""
        state_dim = observation_space.shape
        if hasattr(action_space, 'n'):
            action_dim = action_space.n
        else:
            action_dim = action_space.shape[0]
            
        # ICM looks at unstacked channels (4: RGB + memory)
        icm_state_dim = (state_dim[0] // self.n_stack, state_dim[1], state_dim[2])
        
        print(f"ICM initialized with state_dim: {icm_state_dim}, action_dim: {action_dim}")
            
        self.icm = ICM(icm_state_dim, action_dim, **self.icm_kwargs)
        self.icm.to(self.device)
        
    def train(self) -> None:
        """Override train method to incorporate ICM"""
        if self.icm is None:
            self._init_icm(self.observation_space, self.action_space)
            
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        
        intrinsic_rewards = []
        icm_losses = []
        
        n_steps = len(self.rollout_buffer.actions)
        n_envs = self.rollout_buffer.actions.shape[1]
        n_channels_stacked = self.rollout_buffer.observations.shape[-3]
        n_channels_unstacked = n_channels_stacked // self.n_stack
        for step in range(n_steps - 1):
            obs_stacked = self.rollout_buffer.observations[step]
            next_obs_stacked = self.rollout_buffer.observations[step + 1]
            actions = self.rollout_buffer.actions[step]
            obs = obs_stacked[:, -n_channels_unstacked:, ...]
            next_obs = next_obs_stacked[:, -n_channels_unstacked:, ...]

            obs_tensor = torch.as_tensor(obs, device=self.device)
            next_obs_tensor = torch.as_tensor(next_obs, device=self.device)
            actions_tensor = torch.as_tensor(actions, device=self.device)
            
            intrinsic_reward, icm_loss = self.icm(obs_tensor, next_obs_tensor, actions_tensor)

            intrinsic_rewards.append(intrinsic_reward.cpu().numpy())
            icm_losses.append(icm_loss.item())
            
            self.icm.optimizer.zero_grad()
            icm_loss.backward()
            self.icm.optimizer.step()
        if n_steps > 0 and intrinsic_rewards:
            zero_reward = np.zeros_like(intrinsic_rewards[-1])
            intrinsic_rewards.append(zero_reward)

        intrinsic_rewards_arr = np.array(intrinsic_rewards)
        
        self.rollout_buffer.rewards += intrinsic_rewards_arr * self.intrinsic_reward_scale
        
        self.logger.record("train/icm_loss", np.mean(icm_losses))
        self.logger.record("train/intrinsic_reward_mean", np.mean(intrinsic_rewards_arr))
        self.logger.record("train/intrinsic_reward_std", np.std(intrinsic_rewards_arr))
        self.logger.record("train/intrinsic_reward_scale", self.intrinsic_reward_scale)
        
        super().train()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class AchievementCallback(BaseCallback):
    """
    Tracks Crafter achievements and logs statistics during training.
    Crafter provides achievement info in the info dict after each step.
    """
    def __init__(self, verbose=0):
        super(AchievementCallback, self).__init__(verbose)
        self.episode_achievements = []
        self.episode_scores = []
        self.all_achievements = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        for idx, done in enumerate(dones):
            if done and len(infos) > idx:
                info = infos[idx]
                if 'achievements' in info:
                    achievements = info['achievements']
                    total_achievements = len(achievements)
                    unlocked = sum(1 for v in achievements.values() if v > 0)
                    score_percentage = (unlocked / total_achievements * 100) if total_achievements > 0 else 0
                    
                    self.episode_achievements.append(achievements.copy())
                    self.episode_scores.append(score_percentage)
                    self.logger.record("achievements/score_percentage", score_percentage)
                    self.logger.record("achievements/total_unlocked", unlocked)
                    for key, value in achievements.items():
                        if value > 0:
                            self.logger.record(f"achievements/{key}", value)
                    
                    if self.verbose > 0:
                        print(f"Episode Achievement Score: {score_percentage:.1f}% ({unlocked}/{total_achievements})")
                if 'episode' in info:
                    ep_info = info['episode']
                    if 'r' in ep_info:
                        self.logger.record("rollout/ep_reward", ep_info['r'])
                    if 'l' in ep_info:
                        self.logger.record("rollout/ep_length", ep_info['l'])
        return True

class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Transposes the observation from (H, W, C) to (C, H, W) and
    ensures the observation space is correctly defined for VecFrameStack.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        
        new_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class MapMemoryWrapper(gym.Wrapper):
    """
    Adds a persistent memory map channel that tracks explored regions.
    
    Since we CANNOT access player position from Crafter's internal state,
    we track position heuristically:
    1. Assume starting position is (32, 32) in the 64x64 world grid
    2. Track movement actions (0-4: noop, left, right, up, down)
    3. Update approximate position, clamping to valid bounds
    4. Mark visible region around current position as explored
    
    LIMITATIONS:
    - This tracking is approximate since we don't know if movements fail due to collisions
    - Over long episodes, accumulated errors may cause position drift
    - However, this still provides useful spatial memory for exploration
    
    The memory channel shows which areas have been "visited" (approximately),
    helping the agent avoid revisiting the same locations and encouraging exploration.
    """
    def __init__(self, env, view_radius=4):
        super().__init__(env)
        original_obs_space = env.observation_space

        if not isinstance(original_obs_space, spaces.Box) or original_obs_space.shape != (64, 64, 3):
            raise ValueError(f"MapMemoryWrapper expected observation space shape (64, 64, 3), but got {original_obs_space.shape}")

        # New observation space: (64, 64, 4) with memory channel
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8)

        # Memory grid: 64x64 tracking explored cells
        self.world_size = 64
        self.memory_grid = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        
        # Approximate agent position (will accumulate errors over time)
        self.agent_pos = np.array([32, 32], dtype=np.int32)
        
        # View radius in cells (agent can see ~4-5 cells in each direction)
        self.view_radius = view_radius
        
        # Crafter action space: 0=noop, 1=left, 2=right, 3=up, 4=down, 5+=other
        self.movement_actions = {
            0: (0, 0),    # noop
            1: (-1, 0),   # move left
            2: (1, 0),    # move right  
            3: (0, -1),   # move up
            4: (0, 1),    # move down
        }
        
        print(f"MapMemoryWrapper: Tracking approximate position with {view_radius} cell view radius")
        print("WARNING: Position tracking is heuristic and may drift over long episodes")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_uint8 = (obs * 255.0).astype(np.uint8)
        
        # Reset memory grid and position
        self.memory_grid.fill(0)
        self.agent_pos = np.array([32, 32], dtype=np.int32)
        
        # Mark initial visible region as explored
        self._mark_visible_region()
        memory_channel = self._render_memory_channel()
        new_obs = np.concatenate((obs_uint8, memory_channel), axis=-1)
        return new_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_uint8 = (obs * 255.0).astype(np.uint8)
        
        # Update approximate position based on action
        # Note: This assumes the movement succeeded, which may not always be true
        if action in self.movement_actions:
            dx, dy = self.movement_actions[action]
            self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, 0, self.world_size - 1)
            self.agent_pos[1] = np.clip(self.agent_pos[1] + dy, 0, self.world_size - 1)
        self._mark_visible_region()
        memory_channel = self._render_memory_channel()
        new_obs = np.concatenate((obs_uint8, memory_channel), axis=-1)
        
        return new_obs, reward, terminated, truncated, info
    
    def _mark_visible_region(self):
        """Mark the region around the agent as explored"""
        x, y = self.agent_pos
        
        x_start = max(0, x - self.view_radius)
        x_end = min(self.world_size, x + self.view_radius + 1)
        y_start = max(0, y - self.view_radius)
        y_end = min(self.world_size, y + self.view_radius + 1)
        self.memory_grid[y_start:y_end, x_start:x_end] = 1.0
    
    def _render_memory_channel(self):
        """
        Render the 64x64 memory grid to a 64x64 pixel image.
        Uses 1:1 mapping for simplicity.
        """
        memory_pixels = (self.memory_grid * 255.0).astype(np.uint8)
        return memory_pixels[:, :, np.newaxis]

def make_env(seed: int = 0, use_map_memory: bool = True) -> Callable:
    """
    Utility function for multiprocessed env creation.
    """
    def _init():
        env = crafter.Env(seed=seed)
        env = GymV21CompatibilityV0(env=env)
        env = Monitor(env)
        if use_map_memory:
            env = MapMemoryWrapper(env)
        env = ChannelFirstWrapper(env)
        return env
    return _init
def main(args):
    vec_env_class = SubprocVecEnv if not args.debug else DummyVecEnv
    
    train_env = vec_env_class([make_env(seed=i) for i in range(args.num_envs)])
    train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
    
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    model_path = os.path.join(args.best_model_dir, "best_model.zip")
    
    icm_kwargs = {
        'hidden_dim': args.icm_hidden_dim,
        'eta': args.icm_eta,
        'beta': args.icm_beta,
        'lr': args.icm_lr
    }
    
    if args.continue_training and os.path.exists(model_path):
        print(f"\nLoading existing model from: {model_path}\n")
        model_class = ICM_PPO if args.use_icm else PPO
        model = model_class.load(
            model_path,
            env=train_env,
            custom_objects={"learning_rate": linear_schedule(args.lr)},
            icm_kwargs=icm_kwargs,
            frame_stack=args.frame_stack,
            intrinsic_reward_scale=args.intrinsic_reward_scale
        )
        print(f"Model loaded. Current timesteps: {model.num_timesteps}")
    else:
        print("\nStarting new training from scratch.\n")
        model_class = ICM_PPO if args.use_icm else PPO
        model = model_class(
            "CnnPolicy",
            train_env,
            verbose=1,
            learning_rate=linear_schedule(args.lr),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            tensorboard_log=args.tensorboard_dir,
            icm_kwargs=icm_kwargs,
            frame_stack=args.frame_stack,
            intrinsic_reward_scale=args.intrinsic_reward_scale
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_crafter_Curr"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.best_model_dir,
        log_path=args.eval_log_dir,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    achievement_callback = AchievementCallback(verbose=1)

    total_timesteps = args.total_timesteps
    current_timesteps = model.num_timesteps if args.continue_training and os.path.exists(model_path) else 0
    remaining_timesteps = total_timesteps - current_timesteps

    if remaining_timesteps > 0:
        print(f"\nTraining for {remaining_timesteps:,} more timesteps...\n")
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback, eval_callback, achievement_callback],
            reset_num_timesteps=not (args.continue_training and os.path.exists(model_path))
        )
    else:
        print("\nModel has already been trained for the target number of timesteps.\n")

    final_model_path = os.path.join(args.checkpoint_dir, "ppo_crafter_final")
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for Crafter with ICM and Map Memory.")
    
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps to train the model.")
    parser.add_argument("--continue_training", action="store_true", help="Flag to continue training from the best model.")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments to run.")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv for easier debugging (slower).")
    parser.add_argument("--use_icm", action="store_true", default=True, help="Use Intrinsic Curiosity Module.")

    parser.add_argument("--checkpoint_dir", type=str, default="./NoMap/", help="Directory to save model checkpoints.")
    parser.add_argument("--best_model_dir", type=str, default="./NomapBest/", help="Directory to save the best model.")
    parser.add_argument("--eval_log_dir", type=str, default="./eval_logs/", help="Directory to save evaluation logs.")
    parser.add_argument("--tensorboard_dir", type=str, default="./PPO_crafter_tensorboard/", help="Directory for TensorBoard logs.")

    parser.add_argument("--save_freq", type=int, default=100_000, help="Frequency to save a model checkpoint.")
    parser.add_argument("--eval_freq", type=int, default=50_000, help="Frequency to evaluate the model.")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of episodes to run for evaluation.")
    parser.add_argument("--disable_map_memory", action="store_true", help="Disable the MapMemoryWrapper (no memory channel).")

    parser.add_argument("--frame_stack", type=int, default=4, help="Number of frames to stack for observations.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to optimize the surrogate loss.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for GAE.")
    parser.add_argument("--clip_range", type=float, default=0.25, help="Clipping parameter for PPO.")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient.")

    parser.add_argument("--icm_hidden_dim", type=int, default=256, help="Hidden dimension for ICM networks.")
    parser.add_argument("--icm_eta", type=float, default=0.2, help="ICM intrinsic reward scaling factor.")
    parser.add_argument("--icm_beta", type=float, default=0.2, help="ICM forward vs inverse loss weighting.")
    parser.add_argument("--icm_lr", type=float, default=1e-3, help="ICM learning rate.")
    parser.add_argument("--intrinsic_reward_scale", type=float, default=0.01, help="Scale for intrinsic rewards added to extrinsic rewards.")

    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    main(args)