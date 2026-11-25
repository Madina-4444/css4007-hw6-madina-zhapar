# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

``` text
I used Microsoft Copilot to:
- explain the concepts of Markov Decision Processes (MDPs), value iteration, policies, rewards, and transitions;
- review and verify my manual Part A calculations;
- discuss the implementation strategy for Parts B and C;
- debug Python code and check Gymnasium usage.

All calculations, code execution, experimental results, and final submission materials were produced and verified by me.
```

## Setup and reproduction

Record dependency versions, installation commands and exact run commands. 

### Environment

- Python 3.13.15
- Gymnasium 1.4.0

### Installation

``` text
.\.venv\Scripts\Activate.ps1
pip install gymnasium
pip freeze > requirements.txt
```

## Run commands

``` text
py part_b.py
```

## Output files:

``` text
outputs/sample_episode.txt
outputs/fixed_policy_episode.txt
outputs/episodes.csv
outputs/summary.txt
```

## Part A — Paper calculations

Link your scan/photo here. Include V1 for every nonterminal tile, the three requested V2 calculations, best-action arrows/ties and your explanation. Commit this part before coding. Calculations here: outputs/Part A(1).jpeg, outputs/Part A(2 ).jpeg, outputs/Part A(3).jpeg

### V1 Values

|     |  1  |  2  |  3  |  4  |  
|-----|-----|-----|-----|-----|
|  1  | -1  | -1  |  10 |  G  |
|  2  | -1  |  #  | -1  |  H  |
|  3  | -1  | -1  | -1  | -1  |


### V2 Calculations

Using V1:

- V2(1,2) = 8
- V2(1,3) = 10
- V2(2,3) = 5.21

The full handwritten calculations are included in the submitted scan/photo.

### Best Actions

| State | Best Action |
|-------|-------------|
| (1,2) | Right       |
| (1,3) | Right       |
| (2,3) | Up          |

No action ties occurred.

Two iterations do not guarantee the optimal policy because value information has not fully propagated through the state space. Additional iterations may still change values and actions.

## Part B — FrozenLake

### Environment

Environment: FrozenLake-v1

Map:

SFFF

FHFH

FFFH

HFFG

Python version: 3.13.15

Gymnasium version: 1.4.0

------------------------------------------------------------------------

## 1

``` python
import gymnasium as gym
import os

os.makedirs("outputs", exist_ok=True)

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="ansi"
)

obs, info = env.reset(seed=42)
env.action_space.seed(42)

output = []

output.append("FrozenLake Sample Episode\n")
output.append("========================\n\n")
output.append("START\n")
output.append(env.render())

total_reward = 0
steps = 0

while True:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

    action_name = ["LEFT", "DOWN", "RIGHT", "UP"][action]

    output.append(f"\nStep {steps}: {action_name}\n")
    output.append(env.render())

    if terminated or truncated or steps >= 100:

        if terminated:
            if reward == 1:
                outcome = "GOAL"
            else:
                outcome = "HOLE"
        else:
            outcome = "TIME_LIMIT"

        break

output.append("\n========================\n")
output.append(f"Outcome: {outcome}\n")
output.append(f"Reward: {total_reward}\n")
output.append(f"Steps: {steps}\n")

with open("outputs/sample_episode.txt", "w", encoding="utf-8") as f:
    f.write("".join(output))

env.close()

print("Saved: outputs/sample_episode.txt")
```
## 2
### Fixed policy, written before the experiments

State the action for each nonterminal state and your reasoning. State numbering:

0 1 2 3

4 5 6 7

8 9 10 11

12 13 14 15

Terminal states:

- Goal: 15
- Holes: 5, 7, 11, 12

``` python
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
```

## 3

``` python
def run_episode(policy_type, is_slippery, seed):

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery,
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

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated:
            outcome = "goal" if reward == 1 else "hole"
            break

        if truncated or steps >= 100:
            outcome = "time_limit"
            break

    env.close()

    return {
        "policy": policy_type,
        "slippery": is_slippery,
        "steps": steps,
        "reward": total_reward,
        "outcome": outcome,
    }


for policy in ["random", "fixed"]:
    for slippery in [False, True]:

        result = run_episode(
            policy_type=policy,
            is_slippery=slippery,
            seed=42
        )

        print(result)
```

Output: 
``` text 
(.venv)PS C:\Users\tb071\Desktop\AI\css4007-hw6-madina-zhapar\> py part_b.py 
{'policy': 'random', 'slippery': False, 'steps': 4, 'reward': 0, 'outcome': 'hole'} 
{'policy': 'random', 'slippery': True, 'steps': 20, 'reward': 0, 'outcome': 'hole'} 
{'policy': 'fixed', 'slippery': False, 'steps': 6, 'reward': 1, 'outcome': 'goal'} 
{'policy': 'fixed', 'slippery': True, 'steps': 4, 'reward': 0, 'outcome': 'hole'}
```

## 4

``` python
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
```

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
|---------------------|------------------|-----------------|---------------|---------------|--------------------------|------------------------|------------------------|
| False | random | 11 | 0 | 50 | 0 | 0.00 | 7.98 |
| False | random | 22 | 0 | 50 | 0 | 0.00 | 6.42 |
| False | random | 33 | 1 | 49 | 0 | 2.00 | 6.66 |
| True | random | 11 | 0 | 50 | 0 | 0.00 | 8.30 |
| True | random | 22 | 2 | 48 | 0 | 4.00 | 7.76 |
| True | random | 33 | 0 | 50 | 0 | 0.00 | 6.74 |
| False | fixed | 11 | 50 | 0 | 0 | 100.00 | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100.00 | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100.00 | 6.00 |
| True | fixed | 11 | 6 | 44 | 0 | 12.00 | 5.56 |
| True | fixed | 22 | 1 | 49 | 0 | 2.00 | 4.10 |
| True | fixed | 33 | 2 | 48 | 0 | 4.00 | 4.68 |

Episode-level results are saved in:

- outputs/episodes.csv
- outputs/summary.txt
- outputs/sample_episode.txt
- outputs/fixed_policy_episode.txt

Slipping significantly changed performance. When is_slippery=False, the fixed policy reached the goal in 100% of episodes across all batches. When is_slippery=True, success rates dropped to between 2% and 12% because the agent could move in unintended directions and fall into holes.

Results were generally consistent across batches. Random policies almost always failed regardless of the seed. The fixed policy consistently succeeded without slipping and consistently performed much worse when slipping was enabled.

One successful episode is not sufficient evidence that a policy is effective. The slippery fixed policy occasionally reached the goal, but its overall success rate remained low. Evaluating many episodes provides a more reliable estimate of policy performance.

Successful example trace: - outputs/fixed_policy_episode.txt

Failed example trace: - outputs/sample_episode.txt

These simulations do not train a policy. The fixed policy was specified in advance and never updated based on rewards or outcomes. The experiments only evaluated policy performance under different environment settings.

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

- The solver was first implemented using the exact map and rules from Part A.

Program output: 
### C1 Output:

``` text
States:
[(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)]
```

### check with Ice (2, 3)

Output:

``` text
Transitions from Ice (2,3) going UP:
Probability=0.8, Next=(1,3)
Probability=0.1, Next=(2,3)
Probability=0.1, Next=(2,4)
```

``` text
Initial values:
{(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (2, 1): 0.0, (2, 3): 0.0, (2, 4): 0.0, (3, 1): 0.0, (3, 2): 0.0, (3, 3): 0.0, (3, 4): 0.0}

V1
(1, 1) -1.0
(1, 2) -1.0
(1, 3) 10.0
(1, 4) 0
(2, 1) -1.0
(2, 3) -1.0
(2, 4) 0
(3, 1) -1.0
(3, 2) -1.0
(3, 3) -1.0
(3, 4) -1.0

V2
 Required V2 values
(1,2) = 8.0
(1,3) = 10.0
(2,3) = 5.21
```

Comparison with Part A calculations:

| State | Paper | Program |
|-------|-------|---------|
| (1,2) | 8.00  | 8.00    |
| (1,3) | 10.00 | 10.00   |
| (2,3) | 5.21  | 5.21    |

No discrepancies were found between the paper calculations and the program output.

### Values, policies and convergence

Convergence results:

- Iterations: 12
- Final value change (delta): 2.1652931048166124e-07

The stopping criterion (delta \< 0.000001) was satisfied before reaching the safeguard limit of 10,000 iterations.

Final values:

| State | Value   |
|-------|---------|
| (1,1) | 4.2000  |
| (1,2) | 8.0000  |
| (1,3) | 10.0000 |
| (1,4) | 0       |
| (2,1) | 2.7800  |
| (2,3) | 5.8242  |
| (2,4) | 0       |
| (3,1) | 1.5358  |
| (3,2) | 2.8176  |
| (3,3) | 4.2418  |
| (3,4) | 2.8176  |

Final policy:
| State | Action |
|-------|--------|
| (1,1) | Right  |
| (1,2) | Right  |
| (1,3) | Right  |
| (2,1) | Up     |
| (2,3) | Up     |
| (3,1) | Right  |
| (3,2) | Right  |
| (3,3) | Up     |
| (3,4) | Left   |

Policy map:

→ → → G

↑ \# ↑ H

→ → ↑ ←

Tie-breaking rule:

When multiple actions had the same value, the first action encountered in the action order (UP, RIGHT, DOWN, LEFT) was selected. No action ties occurred in the final policy.

### Simulation results
#### Original map:

|     | 1   | 2   | 3   | 4   |
|-----|-----|-----|-----|-----|
| 1   |  →  |  →  |  →  |  G  |
| 2   |  ↑  | \#  |  ↑  |  H  |
| 3   |  →  |  →  |  ↑  |  ←  |

#### Extended map:

|     | 1   | 2   | 3   | 4   |
|-----|-----|-----|-----|-----|
| 1   |  →  |  →  |  →  |  G  |
| 2   |  ↑  | \#  |  ↑  |  H  |
| 3   |  ↑  |  →  |  ↑  |  ←  |

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
|-------|---------|---------|---------|---------|---------|---------|
| Original | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| Original | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| Original | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| Extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |


### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?
``` text
The addition of sand (D) and water (W) mainly affected the lower part of the map. The optimal action at state (3,1) changed from Right to Up because moving toward the sand tile incurred a larger penalty. Most values near the goal changed only slightly because the goal reward (+10) remained dominant.

Cost mattered in states close to the new floor types. The higher penalties for sand and water reduced the attractiveness of paths that passed through them. Risk mattered at the ice tile because slipping could move the agent into less desirable states or toward the hole.

Simulation results are related to, but not identical to, the computed value function. Value iteration estimates expected infinite-horizon returns, while simulations measure finite episodes using random outcomes. Therefore, empirical mean discounted returns may differ from the computed state values, especially when stochastic ice transitions are involved.
```

### Rule checks

Boundary behaviour:

- Attempting to move outside the map leaves the agent in the same state.
- Example: from (1,1), action UP returns (1,1).

Wall behaviour:

- Attempting to move into a wall (#) leaves the agent in the same state.
- Example: from (2,1), action RIGHT returns (2,1) because of the wall at (2,2).

Ice probabilities:

- Intended direction: 0.8
- Perpendicular direction 1: 0.1
- Perpendicular direction 2: 0.1

Probability check:

0.8 + 0.1 + 0.1 = 1.0

Floor rewards:

- S = -1
- . = -1
- M = -3
- I = -1
- D = -2
- W = -5
- G = +10
- H = -10

Terminal handling:

- Goal (G) and Hole (H) are terminal states.
- No further actions are taken after entering a terminal state.
- No additional rewards are collected after termination.
- Value iteration assigns terminal states a value of 0 after termination.

### Compare the maps

The addition of sand (D) and shallow water (W) changed both state values and the optimal policy in the lower part of the map.

The main policy change occurred at state (3,1):

- Original map: RIGHT
- Extended map: UP

The value at (3,1) also decreased slightly:

- Original map: 1.5358
- Extended map: 1.5020

Floor costs mattered most near the new sand and water states. Because sand has reward -2 and water has reward -5, paths that moved toward these tiles became less attractive. As a result, the optimal policy preferred moving upward rather than moving through the lower row.

Risk mattered at the ice tile (2,3). The optimal policy selected UP from the ice state because it provided the highest expected value despite the possibility of slipping into neighboring states. The policy therefore accounts for stochastic transitions rather than treating movement as deterministic.

Printed value maps and policy maps were used as evidence for the comparison.