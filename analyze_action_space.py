import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read action data
print("Loading action data...")
df = pd.read_csv("action.txt")

print(f"Total actions logged: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Analyze action distributions
print("\n" + "="*80)
print("ACTION SPACE ANALYSIS")
print("="*80)

# Get last 100 episodes for analysis (converged behavior)
last_episodes = df[df['Episode'] > 900]

print(f"\nAnalyzing last 100 episodes (Episodes 901-1000)")
print(f"Total actions in this period: {len(last_episodes)}")

# Alpha distribution (local vs offload)
alpha_stats = last_episodes['Alpha'].describe()
print("\n" + "-"*80)
print("ALPHA (Local Processing Ratio) Distribution:")
print("-"*80)
print(alpha_stats)

# Mu distribution (edge vs neighbor)
mu_stats = last_episodes['Mu'].describe()
print("\n" + "-"*80)
print("MU (Neighbor Offloading Ratio) Distribution:")
print("-"*80)
print(mu_stats)

# Categorize actions
def categorize_action(row):
    alpha = row['Alpha']
    mu = row['Mu']
    
    if alpha > 0.8:
        return "Mostly Local"
    elif alpha < 0.2:
        if mu > 0.5:
            return "Mostly Neighbor"
        else:
            return "Mostly Edge"
    else:
        return "Mixed Strategy"

last_episodes['Strategy'] = last_episodes.apply(categorize_action, axis=1)
strategy_counts = last_episodes['Strategy'].value_counts()

print("\n" + "-"*80)
print("ACTION STRATEGY DISTRIBUTION:")
print("-"*80)
for strategy, count in strategy_counts.items():
    pct = (count / len(last_episodes)) * 100
    print(f"{strategy:<20}: {count:>6} ({pct:>5.1f}%)")

# Check for action collapse (all actions similar)
alpha_std = last_episodes.groupby('Episode')['Alpha'].std().mean()
mu_std = last_episodes.groupby('Episode')['Mu'].std().mean()

print("\n" + "-"*80)
print("ACTION DIVERSITY (Standard Deviation per Episode):")
print("-"*80)
print(f"Alpha Std Dev: {alpha_std:.4f}")
print(f"Mu Std Dev:    {mu_std:.4f}")

if alpha_std < 0.1:
    print("⚠️  WARNING: Alpha has LOW diversity - model may have collapsed!")
if mu_std < 0.1:
    print("⚠️  WARNING: Mu has LOW diversity - model may have collapsed!")

# Analyze by server
print("\n" + "-"*80)
print("ACTIONS BY SERVER:")
print("-"*80)
for server in sorted(last_episodes['Server'].unique()):
    server_data = last_episodes[last_episodes['Server'] == server]
    print(f"\nServer {server}:")
    print(f"  Alpha: {server_data['Alpha'].mean():.3f} ± {server_data['Alpha'].std():.3f}")
    print(f"  Mu:    {server_data['Mu'].mean():.3f} ± {server_data['Mu'].std():.3f}")

# Check for boundary behavior (stuck at 0 or 1)
alpha_at_zero = (last_episodes['Alpha'] < 0.01).sum()
alpha_at_one = (last_episodes['Alpha'] > 0.99).sum()
mu_at_zero = (last_episodes['Mu'] < 0.01).sum()
mu_at_one = (last_episodes['Mu'] > 0.99).sum()

print("\n" + "-"*80)
print("BOUNDARY BEHAVIOR (Actions at extremes):")
print("-"*80)
print(f"Alpha at 0 (<0.01): {alpha_at_zero} ({alpha_at_zero/len(last_episodes)*100:.1f}%)")
print(f"Alpha at 1 (>0.99): {alpha_at_one} ({alpha_at_one/len(last_episodes)*100:.1f}%)")
print(f"Mu at 0 (<0.01):    {mu_at_zero} ({mu_at_zero/len(last_episodes)*100:.1f}%)")
print(f"Mu at 1 (>0.99):    {mu_at_one} ({mu_at_one/len(last_episodes)*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Alpha distribution
axes[0, 0].hist(last_episodes['Alpha'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Alpha Distribution (Last 100 Episodes)', fontweight='bold')
axes[0, 0].set_xlabel('Alpha (Local Processing Ratio)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(last_episodes['Alpha'].mean(), color='red', linestyle='--', label=f'Mean: {last_episodes["Alpha"].mean():.3f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Mu distribution
axes[0, 1].hist(last_episodes['Mu'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Mu Distribution (Last 100 Episodes)', fontweight='bold')
axes[0, 1].set_xlabel('Mu (Neighbor Offloading Ratio)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(last_episodes['Mu'].mean(), color='red', linestyle='--', label=f'Mean: {last_episodes["Mu"].mean():.3f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Alpha vs Mu scatter
axes[0, 2].scatter(last_episodes['Alpha'], last_episodes['Mu'], alpha=0.3, s=1)
axes[0, 2].set_title('Alpha vs Mu (Action Space Coverage)', fontweight='bold')
axes[0, 2].set_xlabel('Alpha')
axes[0, 2].set_ylabel('Mu')
axes[0, 2].grid(True, alpha=0.3)

# Alpha over time
episode_alpha = last_episodes.groupby('Episode')['Alpha'].mean()
axes[1, 0].plot(episode_alpha.index, episode_alpha.values, linewidth=1)
axes[1, 0].set_title('Alpha Over Episodes', fontweight='bold')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Mean Alpha')
axes[1, 0].grid(True, alpha=0.3)

# Mu over time
episode_mu = last_episodes.groupby('Episode')['Mu'].mean()
axes[1, 1].plot(episode_mu.index, episode_mu.values, linewidth=1, color='green')
axes[1, 1].set_title('Mu Over Episodes', fontweight='bold')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Mean Mu')
axes[1, 1].grid(True, alpha=0.3)

# Strategy pie chart
axes[1, 2].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title('Action Strategy Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('action_space_analysis.png', dpi=150)
print("\n" + "="*80)
print("Saved visualization to: action_space_analysis.png")
print("="*80)

# Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if alpha_std < 0.1 and mu_std < 0.1:
    print("🔴 CRITICAL: Action space has COLLAPSED!")
    print("   All agents are taking nearly identical actions.")
    print("   This indicates poor exploration and local optimum.")
elif strategy_counts.iloc[0] / len(last_episodes) > 0.8:
    print("⚠️  WARNING: Model is using ONE dominant strategy >80% of the time")
    print("   This suggests insufficient exploration of the action space.")
elif alpha_at_zero + alpha_at_one > len(last_episodes) * 0.5:
    print("⚠️  WARNING: Alpha is stuck at boundaries (0 or 1) >50% of time")
    print("   Model is not exploring intermediate offloading ratios.")
else:
    print("✅ Action space appears to have reasonable diversity.")

print("\nRECOMMENDATIONS:")
if alpha_std < 0.15:
    print("1. Add entropy bonus to encourage exploration")
    print("2. Increase PPO clip epsilon (0.2 → 0.3)")
    print("3. Use action noise during training")
    print("4. Check if reward signal is too sparse")
