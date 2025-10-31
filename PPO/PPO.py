
import crafter
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback,BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import os

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
                    # Log summary metrics
                    self.logger.record("achievements/score_percentage", score_pct)
                    self.logger.record("achievements/total_unlocked", unlocked)
                    # Optionally log each unlocked achievement count
                    for k, v in achievements.items():
                        if v and v > 0:
                            self.logger.record(f"achievements/{k}", v)
                    if self.verbose:
                        print(f"[AchievementCallback] Episode score: {score_pct:.1f}% ({unlocked}/{total})")
        return True
    
def main():
    # Create the Crafter environment
    env = crafter.Env()
    env = GymV21CompatibilityV0(env=env)

    # Create evaluation environment
    eval_env = crafter.Env()
    eval_env = GymV21CompatibilityV0(env=eval_env)
    eval_env = Monitor(eval_env)


    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,      # Standard PPO learning rate
        n_steps=2048,            # Rollout buffer size (steps per update)
        batch_size=64,           # Minibatch size for policy updates
        n_epochs=10,             # Number of epochs when optimizing the surrogate loss
        gamma=0.99,              # Discount factor
        gae_lambda=0.95,         # Factor for GAE
        clip_range=0.2,          # PPO clipping parameter
        ent_coef=0.01,           # Entropy coefficient to encourage exploration
        tensorboard_log="./PPO_crafter_tensorboard/",  # Enable TensorBoard logging
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps
        save_path="./models/",
        name_prefix="OriginalPPO_Crafter"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./OGPPOModelSave/",
        log_path="./eval_logs/",
        eval_freq=50000,  
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    achievement_callback = AchievementCallback(verbose=1)

    target_timesteps = 2000000  # Set a target number of steps
    current_timesteps = model.num_timesteps
    remaining_timesteps = target_timesteps - current_timesteps

    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps, 
            log_interval=1,
            callback=[checkpoint_callback, eval_callback,achievement_callback],
            reset_num_timesteps=False 
        )
        print(f"\nTraining complete! Total timesteps: {model.num_timesteps:,}")


    # Save the final model
    model.save("Original_PPO_crafter_final")
    print("Model saved as Original_PPO_crafter_final")

    # Load and test the model
    model = PPO.load("Original_PPO_crafter_final")
    print("Testing trained model...")

    # Create new env for testing
    test_env = crafter.Env()
    test_env = crafter.Recorder(
        test_env, './OriginalPPO_crafter_logs',
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    # Wrap for gymnasium compatibility AFTER recorder
    test_env = GymV21CompatibilityV0(env=test_env)

    obs, info = test_env.reset()
    episode_reward = 0
    episode_length = 0

    for _ in range(1000):  # Test for 1000 steps
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

if __name__ == "__main__":
    main()