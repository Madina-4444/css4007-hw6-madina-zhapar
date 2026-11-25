import gymnasium as gym
import csv
import os

os.makedirs("outputs", exist_ok=True)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

fixed_policy = {
    0: RIGHT,
    1: RIGHT,
    2: DOWN,
    3: DOWN,
    4: DOWN,
    6: DOWN,
    8: RIGHT,
    9: RIGHT,
    10: DOWN,
    13: RIGHT,
    14: RIGHT,
}


def run_episode(policy_type, is_slippery, seed):

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery
    )

    state, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    steps = 0
    total_reward = 0

    while True:

        if policy_type == "random":
            action = env.action_space.sample()
        else:
            action = fixed_policy.get(state, RIGHT)

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        state = next_state

        if terminated:

            if state == 15:
                outcome = "goal"
            else:
                outcome = "hole"

            break

        if truncated or steps >= 100:
            outcome = "time_limit"
            break

    env.close()

    return {
        "steps": steps,
        "reward": total_reward,
        "outcome": outcome
    }


results = []

conditions = [
    ("random", False),
    ("random", True),
    ("fixed", False),
    ("fixed", True)
]

batch_seeds = [11, 22, 33]

for policy, slippery in conditions:

    for batch_seed in batch_seeds:

        for i in range(50):

            episode_seed = 1000 * batch_seed + i

            result = run_episode(
                policy_type=policy,
                is_slippery=slippery,
                seed=episode_seed
            )

            results.append({
                "condition": f"{policy}_{slippery}",
                "batch_seed": batch_seed,
                "episode_seed": episode_seed,
                "policy": policy,
                "steps": result["steps"],
                "total_reward": result["reward"],
                "outcome": result["outcome"]
            })


with open(
    "outputs/episodes.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "batch_seed",
            "episode_seed",
            "policy",
            "steps",
            "total_reward",
            "outcome"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


with open("outputs/summary.txt", "w", encoding="utf-8") as f:

    for policy, slippery in conditions:

        for batch_seed in batch_seeds:

            batch = [
                r for r in results
                if r["policy"] == policy
                and r["batch_seed"] == batch_seed
                and r["condition"] == f"{policy}_{slippery}"
            ]

            goals = sum(1 for r in batch if r["outcome"] == "goal")
            holes = sum(1 for r in batch if r["outcome"] == "hole")
            time_limits = sum(
                1 for r in batch
                if r["outcome"] == "time_limit"
            )

            mean_steps = (
                sum(r["steps"] for r in batch)
                / len(batch)
            )

            success_rate = goals / len(batch) * 100

            f.write(
                f"Slippery={slippery}, "
                f"Policy={policy}, "
                f"Batch={batch_seed}, "
                f"Goals={goals}/50, "
                f"Holes={holes}/50, "
                f"TimeLimits={time_limits}/50, "
                f"Success={success_rate:.2f}%, "
                f"MeanSteps={mean_steps:.2f}\n"
            )

print("Saved outputs/episodes.csv")
print("Saved outputs/summary.txt")
print(f"Total episodes: {len(results)}")