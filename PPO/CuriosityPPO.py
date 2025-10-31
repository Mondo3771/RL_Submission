import crafter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback , BaseCallback
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
from gym import spaces

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, eta=0.2, beta=0.2, lr=1e-3):
        super(ICM, self).__init__()
        self.eta = eta
        self.beta = beta
        self.action_dim = action_dim
        
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
        batch_size = state.shape[0]
        
        state_feat = self.feature_net(state)
        next_state_feat = self.feature_net(next_state)

        if action is None:
            raise ValueError("Action tensor is None")
        if action.dim() > 1:
            action_indices = action.argmax(dim=-1).view(-1)
        else:
            action_indices = action.view(-1).long()
        action_one_hot = F.one_hot(action_indices.long(), num_classes=self.action_dim).float().to(action.device)

        pred_action_logits = self.inverse_net(torch.cat([state_feat, next_state_feat], dim=1))
        
        if action.dim() == 1:
            inverse_loss = F.cross_entropy(pred_action_logits, action.long())
        else:
            inverse_loss = F.cross_entropy(pred_action_logits, action.argmax(dim=1))
        forward_input = torch.cat([state_feat, action_one_hot], dim=1)
        pred_next_state_feat = self.forward_net(forward_input)
        per_element_loss = F.mse_loss(pred_next_state_feat, next_state_feat, reduction='none')  # (batch, feat_dim)
        forward_loss_per_sample = per_element_loss.mean(dim=1)  # (batch,)
        forward_loss = forward_loss_per_sample.mean() 
        intrinsic_reward = self.eta * forward_loss_per_sample.detach()
        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        return intrinsic_reward, total_loss
    

class ICM_PPO(PPO):
    """
    PPO with Intrinsic Curiosity Module
    """
    def __init__(self, *args, **kwargs):
        self.icm = None
        self.icm_kwargs = kwargs.pop('icm_kwargs', {})
        super(ICM_PPO, self).__init__(*args, **kwargs)
        
    def _init_icm(self, observation_space, action_space):
        """Initialize ICM module"""
        state_dim = observation_space.shape
        if hasattr(action_space, 'n'):
            action_dim = action_space.n
        else:
            action_dim = action_space.shape[0]
            
        self.icm = ICM(state_dim, action_dim, **self.icm_kwargs)
        self.icm.to(self.device)
        
    def train(self) -> None:
        """
        Override train method to incorporate ICM
        """
        if self.icm is None:
            self._init_icm(self.observation_space, self.action_space)
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        intrinsic_rewards = []
        icm_losses = []
        n_steps = len(self.rollout_buffer.actions)
        batch_size = self.rollout_buffer.observations.shape[1] 
        
        for step in range(n_steps):
            # Get observations and actions for all environments at this step
            obs = self.rollout_buffer.observations[step]  # Shape: (num_envs, channels, height, width)
            next_obs = self.rollout_buffer.observations[step + 1] if step < n_steps - 1 else obs
            actions = self.rollout_buffer.actions[step]  # Shape: (num_envs,)
            
            # Convert to tensors and ensure correct shape
            obs_tensor = torch.as_tensor(obs, device=self.device).float()
            next_obs_tensor = torch.as_tensor(next_obs, device=self.device).float()
            actions_tensor = torch.as_tensor(actions, device=self.device)
            
            # Reshape if needed for frame stacking
            if obs_tensor.dim() == 5:  # (num_envs, frame_stack, channels, height, width)
                obs_tensor = obs_tensor.view(-1, *obs_tensor.shape[2:])
                next_obs_tensor = next_obs_tensor.view(-1, *next_obs_tensor.shape[2:])
                actions_tensor = actions_tensor.view(-1)
            
            # Get intrinsic reward and ICM loss
            intrinsic_reward, icm_loss = self.icm(obs_tensor, next_obs_tensor, actions_tensor)
  
            # Reshape intrinsic reward back to per-environment
            if intrinsic_reward.shape[0] == batch_size * self.rollout_buffer.observations.shape[2]:  # frame_stack
                intrinsic_reward = intrinsic_reward.view(batch_size, -1).mean(dim=1)
            
            intrinsic_rewards.append(intrinsic_reward.cpu().numpy())
            icm_losses.append(icm_loss.item())
            
            # Update ICM
            self.icm.optimizer.zero_grad()
            icm_loss.backward()
            self.icm.optimizer.step()
            
        # Add intrinsic rewards to extrinsic rewards (scaled appropriately)
        intrinsic_rewards = np.array(intrinsic_rewards)  # Shape: (n_steps, num_envs)
        intrinsic_reward_scale = 0.01  # Scale factor for intrinsic rewards
        
        # Add to rewards for each step and environment
        for step in range(n_steps):
            self.rollout_buffer.rewards[step] += intrinsic_rewards[step] * intrinsic_reward_scale
        
        # Log ICM metrics
        self.logger.record("train/icm_loss", np.mean(icm_losses))
        self.logger.record("train/intrinsic_reward_mean", np.mean(intrinsic_rewards))
        self.logger.record("train/intrinsic_reward_std", np.std(intrinsic_rewards))
        
        # Continue with normal PPO training using the parent class method
        super().train()

class AchievementCallback(BaseCallback):
    """
    Logs achievement statistics (from env info['achievements']) to tensorboard.
    Expects 'achievements' in the info dict as a mapping achievement_name -> count.
    """
    def __init__(self, verbose=0):
        super(AchievementCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for idx, done in enumerate(dones):
            if done and idx < len(infos):
                info = infos[idx]
                if not isinstance(info, dict):
                    continue
                if "achievements" in info and isinstance(info["achievements"], dict):
                    achievements = info["achievements"]
                    total = len(achievements)
                    unlocked = sum(1 for v in achievements.values() if v and v > 0)
                    score_pct = (unlocked / total * 100.0) if total > 0 else 0.0
                    self.logger.record("achievements/score_percentage", score_pct)
                    self.logger.record("achievements/total_unlocked", unlocked)
                    for k, v in achievements.items():
                        if v and v > 0:
                            self.logger.record(f"achievements/{k}", v)
                    if self.verbose:
                        print(f"[AchievementCallback] Episode score: {score_pct:.1f}% ({unlocked}/{total})")
        return True

def make_env(seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env creation.
    """
    def _init():
        env = crafter.Env(seed=seed)
        env = GymV21CompatibilityV0(env=env)
        env = Monitor(env)
        return env
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will go from 1.0 to 0.0.
        """
        return progress_remaining * initial_value
    return func

def main(args):
    print("==============================================")
    print("        PPO + ICM Crafter Training Run        ")
    print("==============================================")
    print(f"Using {args.num_envs} parallel environments.")
    print(f"ICM enabled: {args.use_icm}")
    
    # Use SubprocVecEnv for parallel environments, which is much faster
    # Use DummyVecEnv for debugging (runs sequentially)
    vec_env_class = SubprocVecEnv if not args.debug else DummyVecEnv
    
    # Create a vectorized environment for training
    train_env = vec_env_class([make_env(seed=i) for i in range(args.num_envs)])
    # Apply the Frame Stacking wrapper
    train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
    
    # Create a single environment for evaluation, then wrap it to be a VecEnv
    # This is the standard way to prepare an eval env for EvalCallback
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)
    model_path = os.path.join(args.best_model_dir, "best_model.zip")
    
    # ICM parameters
    icm_kwargs = {
        'hidden_dim': args.icm_hidden_dim,
        'eta': args.icm_eta,
        'beta': args.icm_beta,
        'lr': args.icm_lr
    }
    
    if args.continue_training and os.path.exists(model_path):
        print(f"\nLoading existing model from: {model_path}\n")
        if args.use_icm:
            model = ICM_PPO.load(
                model_path,
                env=train_env,
                custom_objects={"learning_rate": linear_schedule(args.lr)},
                icm_kwargs=icm_kwargs
            )
        else:
            model = PPO.load(
                model_path,
                env=train_env,
                custom_objects={"learning_rate": linear_schedule(args.lr)},
            )
        print(f"Model loaded. Current timesteps: {model.num_timesteps}")
    else:
        print("\nStarting new training from scratch.\n")
        if args.use_icm:
            model = ICM_PPO(
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
                icm_kwargs=icm_kwargs
            )
        else:
            model = PPO(
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
    # If continuing, the model's internal counter is used. Otherwise, start from 0.
    current_timesteps = model.num_timesteps if args.continue_training and os.path.exists(model_path) else 0
    remaining_timesteps = total_timesteps - current_timesteps

    if remaining_timesteps > 0:
        print(f"\nTraining for {remaining_timesteps:,} more timesteps...\n")
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback, eval_callback,achievement_callback],
            reset_num_timesteps=not (args.continue_training and os.path.exists(model_path))
        )
    else:
        print("\nModel has already been trained for the target number of timesteps.\n")

    final_model_path = os.path.join(args.checkpoint_dir, "ppo_crafter_final")
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for Crafter.")
    
    # --- Core Training Arguments ---
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total timesteps to train the model.")
    parser.add_argument("--continue_training", action="store_true", help="Flag to continue training from the best model.")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments to run.")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv for easier debugging (slower).")
    parser.add_argument("--use_icm", action="store_true", default=True, help="Use Intrinsic Curiosity Module.")

    # --- Directory Arguments ---
    parser.add_argument("--checkpoint_dir", type=str, default="./models/", help="Directory to save model checkpoints.")
    parser.add_argument("--best_model_dir", type=str, default="./best_model/", help="Directory to save the best model.")
    parser.add_argument("--eval_log_dir", type=str, default="./eval_logs/", help="Directory to save evaluation logs.")
    parser.add_argument("--tensorboard_dir", type=str, default="./PPO_crafter_tensorboard/", help="Directory for TensorBoard logs.")

    # --- Callback Arguments ---
    parser.add_argument("--save_freq", type=int, default=100_000, help="Frequency to save a model checkpoint.")
    parser.add_argument("--eval_freq", type=int, default=50_000, help="Frequency to evaluate the model.")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of episodes to run for evaluation.")

    # --- PPO & Environment Hyperparameters ---
    parser.add_argument("--frame_stack", type=int, default=4, help="Number of frames to stack for observations.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--n_epochs", type=int, default=4, help="Number of epochs to optimize the surrogate loss.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for GAE.")
    parser.add_argument("--clip_range", type=float, default=0.1, help="Clipping parameter for PPO.")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient.")

    # --- ICM Hyperparameters ---
    parser.add_argument("--icm_hidden_dim", type=int, default=256, help="Hidden dimension for ICM networks.")
    parser.add_argument("--icm_eta", type=float, default=0.2, help="ICM intrinsic reward scaling factor.")
    parser.add_argument("--icm_beta", type=float, default=0.2, help="ICM forward vs inverse loss weighting.")
    parser.add_argument("--icm_lr", type=float, default=1e-3, help="ICM learning rate.")

    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    main(args)