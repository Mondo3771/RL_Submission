import json
import matplotlib.pyplot as plt
import numpy as np
import os


ALIGN_TO_ZERO = True 
CLIP_MAX_TIMESTEPS = 1000000 

# Load metrics from all four methods
methods = {
    'Baseline': 'metrics_baseline.json',
    'Dueling DQN': 'metrics_dueling.json',
    'PER': 'metrics_per.json',
    'Reward Shaping': 'metrics_shaped.json'
}

data = {}
for method_name, filename in methods.items():
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data[method_name] = json.load(f)
        print(f"✅ Loaded {method_name}: {len(data[method_name]['timesteps'])} data points")
    else:
        print(f"⚠️  {filename} not found, skipping {method_name}")

if not data:
    print("❌ No metrics files found! Run training first.")
    exit(1)

for method_name, method_data in data.items():
    timesteps = np.array(method_data['timesteps'])

    if ALIGN_TO_ZERO and len(timesteps) > 0:
        offset = timesteps[0]
        timesteps = timesteps - offset
        print(f"  Shifted {method_name} by {offset:,} steps (now starts at 0)")
    
    # Clip to max timesteps
    if CLIP_MAX_TIMESTEPS is not None:
        mask = timesteps <= CLIP_MAX_TIMESTEPS
        if not np.all(mask):
            original_len = len(timesteps)
            timesteps = timesteps[mask]
            method_data['timesteps'] = timesteps.tolist()
            method_data['geometric_means'] = [method_data['geometric_means'][i] for i in range(len(mask)) if mask[i]]
            method_data['eval_rewards'] = [method_data['eval_rewards'][i] for i in range(len(mask)) if mask[i]]
            method_data['survival_times'] = [method_data['survival_times'][i] for i in range(len(mask)) if mask[i]]
            method_data['cumulative_rewards'] = [method_data['cumulative_rewards'][i] for i in range(len(mask)) if mask[i]]
            method_data['achievement_rates'] = [method_data['achievement_rates'][i] for i in range(len(mask)) if mask[i]]
            print(f"  Clipped {method_name} from {original_len} to {len(timesteps)} points (max {CLIP_MAX_TIMESTEPS:,} steps)")
        else:
            method_data['timesteps'] = timesteps.tolist()
    else:
        method_data['timesteps'] = timesteps.tolist()

print()


# FIGURE 1: Main Metrics (2x2 grid)

fig1 = plt.figure(figsize=(16, 12))
gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

colors = {
    'Baseline': 'purple',
    'Dueling DQN': 'blue',
    'PER': 'green',
    'Reward Shaping': 'orange'
}

# Plot 1: Geometric Mean of Achievement Unlock Rates (top-left)
ax1 = fig1.add_subplot(gs1[0, 0])
for method_name, method_data in data.items():
    ax1.plot(method_data['timesteps'], method_data['geometric_means'], 
             linewidth=2.5, alpha=0.8, label=method_name, color=colors[method_name])
ax1.set_xlabel('Timesteps', fontsize=13)
ax1.set_ylabel('Geometric Mean Score', fontsize=13)
ax1.set_title('Geometric Mean of Achievement Unlock Rates', fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Survival Time (top-right)
ax2 = fig1.add_subplot(gs1[0, 1])
for method_name, method_data in data.items():
    ax2.plot(method_data['timesteps'], method_data['survival_times'], 
             linewidth=2.5, alpha=0.8, label=method_name, color=colors[method_name])
ax2.set_xlabel('Timesteps', fontsize=13)
ax2.set_ylabel('Avg Episode Length (steps)', fontsize=13)
ax2.set_title('Survival Time', fontsize=15, fontweight='bold')
ax2.legend(loc='best', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Cumulative Reward (bottom-left)
ax3 = fig1.add_subplot(gs1[1, 0])
for method_name, method_data in data.items():
    ax3.plot(method_data['timesteps'], method_data['cumulative_rewards'], 
             linewidth=2.5, alpha=0.8, label=method_name, color=colors[method_name])
ax3.set_xlabel('Timesteps', fontsize=13)
ax3.set_ylabel('Avg Cumulative Reward', fontsize=13)
ax3.set_title('Cumulative Reward per Episode', fontsize=15, fontweight='bold')
ax3.legend(loc='best', fontsize=12)
ax3.grid(True, alpha=0.3)

# Plot 4: Evaluation Reward (bottom-right)
ax4 = fig1.add_subplot(gs1[1, 1])
for method_name, method_data in data.items():
    ax4.plot(method_data['timesteps'], method_data['eval_rewards'], 
             linewidth=2.5, alpha=0.8, label=method_name, color=colors[method_name])
ax4.set_xlabel('Timesteps', fontsize=13)
ax4.set_ylabel('Mean Eval Reward', fontsize=13)
ax4.set_title('Evaluation Reward', fontsize=15, fontweight='bold')
ax4.legend(loc='best', fontsize=12)
ax4.grid(True, alpha=0.3)

plt.suptitle('Crafter Training Comparison - Main Metrics', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('crafter_metrics_main.png', dpi=300, bbox_inches='tight')
print("Main metrics plot saved as 'crafter_metrics_main.png'")


# FIGURE 2: Achievement Unlock Rates (separate figure)

# Find most common achievements across all methods
achievement_names = set()
for method_data in data.values():
    for achievement_dict in method_data['achievement_rates']:
        achievement_names.update(achievement_dict.keys())

# Count occurrences of each achievement
all_achievements_count = {}
for method_data in data.values():
    for achievement_dict in method_data['achievement_rates']:
        for achievement in achievement_dict.keys():
            all_achievements_count[achievement] = all_achievements_count.get(achievement, 0) + 1

if all_achievements_count:
    # Get top 6 most common achievements
    top_achievements = sorted(all_achievements_count.items(), key=lambda x: x[1], reverse=True)[:6]
    
    fig2 = plt.figure(figsize=(18, 10))
    gs2 = fig2.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    for idx, (achievement, count) in enumerate(top_achievements):
        row = idx // 3
        col = idx % 3
        ax = fig2.add_subplot(gs2[row, col])
        
        for method_name, method_data in data.items():
            rates = []
            for achievement_dict in method_data['achievement_rates']:
                rates.append(achievement_dict.get(achievement, 0.0))
            
            if any(r > 0 for r in rates):
                ax.plot(method_data['timesteps'], rates, 
                       linewidth=2.5, alpha=0.8, label=method_name, color=colors[method_name])
        
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Unlock Rate (%)', fontsize=12)
        ax.set_title(f'{achievement.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Achievement Unlock Rates - Top 6 Achievements', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('crafter_achievements.png', dpi=300, bbox_inches='tight')
    print("Achievement rates plot saved as 'crafter_achievements.png'")
else:
    print("No achievements tracked")


# Print final statistics

print("\n" + "="*80)
print("FINAL COMPARISON STATISTICS")
print("="*80)

for method_name, method_data in data.items():
    if method_data['timesteps']:
        print(f"\n{method_name}:")
        print(f"  Final Timesteps: {method_data['timesteps'][-1]:,}")
        print(f"  Final Geometric Mean: {method_data['geometric_means'][-1]:.2f}")
        print(f"  Best Geometric Mean: {max(method_data['geometric_means']):.2f}")
        print(f"  Final Survival Time: {method_data['survival_times'][-1]:.1f} steps")
        print(f"  Final Cumulative Reward: {method_data['cumulative_rewards'][-1]:.2f}")
        print(f"  Final Eval Reward: {method_data['eval_rewards'][-1]:.2f}")
        print(f"  Best Eval Reward: {max(method_data['eval_rewards']):.2f}")

print("="*80)

plt.show()