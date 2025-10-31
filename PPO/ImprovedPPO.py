import crafter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import os
import argparse
from typing import Callable
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

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
    # Use SubprocVecEnv for parallel environments, which is much faster
    vec_env_class = SubprocVecEnv 
    
    # Create a vectorized environment for training
    train_env = vec_env_class([make_env(seed=i) for i in range(args.num_envs)])
    # Apply the Frame Stacking wrapper
    train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
    
    # Create a single environment for evaluation, then wrap it to be a VecEnv
    # This is the standard way to prepare an eval env for EvalCallback
    eval_env = DummyVecEnv([make_env()])
    # Now apply the Frame Stacking wrapper to the vectorized eval_env
    eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    model_path = os.path.join(args.best_model_dir, "best_model.zip")
    if args.continue_training and os.path.exists(model_path):
        print(f"\nLoading existing model from: {model_path}\n")
        model = PPO.load(
            model_path,
            env=train_env,
            custom_objects={"learning_rate": linear_schedule(args.lr)},
        )
        print(f"Model loaded. Current timesteps: {model.num_timesteps}")
    else:
        print("\nStarting new training from scratch.\n")
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
        name_prefix="ppo_crafter"
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
    final_model_path = os.path.join(args.checkpoint_dir, "ImprovedPPO_Crafter")
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for Crafter.")
    
    # --- Core Training Arguments ---
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total timesteps to train the model.")
    parser.add_argument("--continue_training", action="store_true", help="Flag to continue training from the best model.")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments to run.")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv for easier debugging (slower).")

    # --- Directory Arguments ---
    parser.add_argument("--checkpoint_dir", type=str, default="./ImporvedPPOModel/", help="Directory to save model checkpoints.")
    parser.add_argument("--best_model_dir", type=str, default="./IPPObest_model/", help="Directory to save the best model.")
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

    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    main(args)
