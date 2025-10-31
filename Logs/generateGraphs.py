import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
LOG_FILES = {
    "OriginalPPO": "OriginalPPO.log",
    "ImprovedPPO": "ImprovedPPO.log",
    "CuriosityMapICM": "CuriosityWithMapMemoryPPO.log",
    "Curiosity": "CuriosityPPO.log"
}
OUTPUT_DIR = "plots"

# --- Helper to parse logs ---
def parse_log(filepath):
    data = {
        "timesteps": [],
        "ep_rew_mean": [],
        "ep_len_mean": [],
        "score_percentage": [],
    }
    achievements = {}

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_timestep = None
    for line in lines:
        # total_timesteps tracking
        if "total_timesteps" in line:
            match = re.search(r"total_timesteps\s+\|\s+([0-9]+)", line)
            if match:
                current_timestep = int(match.group(1))

        # rollout metrics
        if "ep_rew_mean" in line:
            val = float(re.search(r"ep_rew_mean\s+\|\s+([\-0-9.]+)", line).group(1))
            data["ep_rew_mean"].append(val)
            data["timesteps"].append(current_timestep if current_timestep else len(data["timesteps"]))
        if "ep_len_mean" in line:
            val = float(re.search(r"ep_len_mean\s+\|\s+([\-0-9.]+)", line).group(1))
            data["ep_len_mean"].append(val)
        if "score_percentage" in line:
            val = float(re.search(r"score_percentage\s+\|\s+([\-0-9.]+)", line).group(1))
            data["score_percentage"].append(val)

        # achievements (only in certain logs)
        ach_match = re.findall(r"\|\s+([a-z_]+)\s+\|\s+([\-0-9.]+)", line)
        for a_name, a_val in ach_match:
            if a_name in [
                "collect_drink", "collect_sapling", "collect_wood", "place_plant",
                "place_table", "wake_up", "defeat_zombie", "make_wood_pickaxe",
                "make_wood_sword", "eat_cow"
            ]:
                achievements.setdefault(a_name, []).append(float(a_val))

    # --- Fix uneven column lengths (fill shorter lists with NaN) ---
    max_len = max(len(v) for v in data.values() if len(v) > 0)
    for key in data:
        if len(data[key]) < max_len:
            data[key].extend([None] * (max_len - len(data[key])))

    df = pd.DataFrame(data)
    return df, achievements


# --- Plotting functions ---
def plot_metric(all_data, metric, ylabel, subfolder):
    os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)
    plt.figure(figsize=(8, 5))

    for label, df in all_data.items():
        if metric in df.columns and not df[metric].dropna().empty:
            plt.plot(df["timesteps"], df[metric], label=label)

    if plt.gca().has_data():
        plt.xlabel("Timesteps")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Comparison")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, subfolder, f"{metric}_comparison.png"))
    plt.close()


def plot_achievements(all_achievements):
    os.makedirs(os.path.join(OUTPUT_DIR, "achievements"), exist_ok=True)
    all_keys = set(k for d in all_achievements.values() for k in d.keys())

    if not all_keys:
        print("ℹ️ No achievements found in any log.")
        return

    for ach in all_keys:
        plt.figure(figsize=(8, 5))
        for label, ach_dict in all_achievements.items():
            if ach in ach_dict and len(ach_dict[ach]) > 0:
                plt.plot(range(len(ach_dict[ach])), ach_dict[ach], label=label)
        if plt.gca().has_data():
            plt.xlabel("Update Step")
            plt.ylabel(f"{ach} count")
            plt.title(f"Achievement: {ach}")
            plt.legend()
            plt.tight_layout()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(OUTPUT_DIR, "achievements", f"{ach}.png"))
        plt.close()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    all_data = {}
    all_achievements = {}

    for name, path in LOG_FILES.items():
        if os.path.exists(path):
            print(f"📊 Parsing {path}...")
            df, ach = parse_log(path)
            all_data[name] = df
            all_achievements[name] = ach
        else:
            print(f"⚠️ File not found: {path}")

    # Plot available metrics
    plot_metric(all_data, "ep_rew_mean", "Average Episode Reward", "rewards")
    plot_metric(all_data, "ep_len_mean", "Average Episode Length", "episode_length")
    plot_metric(all_data, "score_percentage", "Achievement Score (%)", "achievements")

    # Plot achievements only for logs that have them
    plot_achievements(all_achievements)

    print("✅ All available plots generated under 'plots/'")
