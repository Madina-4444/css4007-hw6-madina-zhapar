import random
import csv
import os

os.makedirs("outputs", exist_ok=True)

GAMMA = 0.9
BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
STEP_LIMIT = 100

ORIGINAL_GRID = [
    [".", "M", ".", "G"],
    [".", "#", "I", "H"],
    ["S", ".", ".", "."]
]

EXTENDED_GRID = [
    [".", "M", ".", "G"],
    [".", "#", "I", "H"],
    ["S", "D", ".", "W"]
]

TILE_REWARDS = {
    "S": -1,
    ".": -1,
    "M": -3,
    "I": -1,
    "D": -2,
    "W": -5,
    "G": 10,
    "H": -10
}

ACTIONS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1)
}

def get_tile(grid, row, col):
    return grid[row - 1][col - 1]


def is_terminal(grid, row, col):
    return get_tile(grid, row, col) in ["G", "H"]


def in_bounds(grid, row, col):
    rows = len(grid)
    cols = len(grid[0])

    return 1 <= row <= rows and 1 <= col <= cols


def move(grid, row, col, action):

    dr, dc = ACTIONS[action]

    new_row = row + dr
    new_col = col + dc

    if not in_bounds(grid, new_row, new_col):
        return row, col

    if get_tile(grid, new_row, new_col) == "#":
        return row, col

    return new_row, new_col


def get_reward(grid, row, col):
    return TILE_REWARDS[get_tile(grid, row, col)]

def build_states(grid):

    states = []

    rows = len(grid)
    cols = len(grid[0])

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):

            if get_tile(grid, row, col) != "#":
                states.append((row, col))

    return states

def get_perpendicular_actions(action):

    if action in ["UP", "DOWN"]:
        return ["LEFT", "RIGHT"]

    return ["UP", "DOWN"]

def get_transitions(grid, row, col, action):

    tile = get_tile(grid, row, col)

    if tile != "I":

        next_row, next_col = move(
            grid,
            row,
            col,
            action
        )

        return [
            (1.0, next_row, next_col)
        ]

    transitions = []

    main_row, main_col = move(
        grid,
        row,
        col,
        action
    )

    transitions.append(
        (0.8, main_row, main_col)
    )

    for slip_action in get_perpendicular_actions(action):

        slip_row, slip_col = move(
            grid,
            row,
            col,
            slip_action
        )

        transitions.append(
            (0.1, slip_row, slip_col)
        )

    return transitions

def one_value_iteration(grid, states, previous_values):

    new_values = {}

    for state in states:

        row, col = state

        if is_terminal(grid, row, col):
            new_values[state] = 0
            continue

        action_values = []

        for action in ACTIONS:

            q = 0

            for prob, next_row, next_col in get_transitions(
                grid,
                row,
                col,
                action
            ):

                reward = get_reward(
                    grid,
                    next_row,
                    next_col
                )

                q += prob * (
                    reward +
                    GAMMA *
                    previous_values[(next_row, next_col)]
                )

            action_values.append(q)

        new_values[state] = max(action_values)

    return new_values

def value_iteration_until_convergence(
    grid,
    states
):

    values = {}

    for state in states:
        values[state] = 0.0

    iteration = 0
    delta = float("inf")

    while delta > 0.000001 and iteration < 10000:

        new_values = one_value_iteration(
            grid,
            states,
            values
        )

        delta = 0

        for state in states:

            change = abs(
                new_values[state] -
                values[state]
            )

            delta = max(delta, change)

        values = new_values
        iteration += 1

    return values, iteration, delta

def extract_policy(
    grid,
    states,
    values
):

    policy = {}

    for state in states:

        row, col = state

        if is_terminal(grid, row, col):
            policy[state] = "*"
            continue

        best_action = None
        best_value = float("-inf")

        for action in ACTIONS:

            q = 0

            for prob, next_row, next_col in get_transitions(
                grid,
                row,
                col,
                action
            ):

                reward = get_reward(
                    grid,
                    next_row,
                    next_col
                )

                q += prob * (
                    reward +
                    GAMMA *
                    values[(next_row, next_col)]
                )

            if q > best_value:
                best_value = q
                best_action = action

        policy[state] = best_action

    return policy

def build_policy(grid):

    states = build_states(grid)

    values, iterations, delta = (
        value_iteration_until_convergence(
            grid,
            states
        )
    )

    policy = extract_policy(
        grid,
        states,
        values
    )

    return states, values, policy

original_states, original_values, original_policy = build_policy(ORIGINAL_GRID)

extended_states, extended_values, extended_policy = build_policy(EXTENDED_GRID)

print("\nORIGINAL POLICY")
print(original_policy)

print("\nEXTENDED POLICY")
print(extended_policy)

def find_start(grid):

    rows = len(grid)
    cols = len(grid[0])

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):

            if get_tile(grid, row, col) == "S":
                return row, col

def simulate_episode(
    grid,
    policy,
    seed
):

    random.seed(seed)

    row, col = find_start(grid)

    steps = 0

    discounted_return = 0.0

    discount = 1.0

    while steps < STEP_LIMIT:

        state = (row, col)

        if is_terminal(grid, row, col):

            if get_tile(grid, row, col) == "G":
                outcome = "goal"
            else:
                outcome = "hole"

            return {
                "outcome": outcome,
                "steps": steps,
                "discounted_return": discounted_return
            }

        action = policy[state]

        transitions = get_transitions(
            grid,
            row,
            col,
            action
        )

        r = random.random()

        cumulative = 0.0

        for prob, next_row, next_col in transitions:

            cumulative += prob

            if r <= cumulative:

                reward = get_reward(
                    grid,
                    next_row,
                    next_col
                )

                discounted_return += (
                    discount * reward
                )

                row = next_row
                col = next_col

                break

        discount *= GAMMA

        steps += 1

    return {
        "outcome": "time_limit",
        "steps": steps,
        "discounted_return": discounted_return
    }

test = simulate_episode(
    ORIGINAL_GRID,
    original_policy,
    11000
)

print("\nTEST EPISODE")
print(test)

def run_batch(grid, policy, batch_seed):

    goals = 0
    holes = 0
    time_limits = 0

    total_steps = 0
    total_return = 0

    for i in range(50):

        episode_seed = 1000 * batch_seed + i

        result = simulate_episode(
            grid,
            policy,
            episode_seed
        )

        if result["outcome"] == "goal":
            goals += 1

        elif result["outcome"] == "hole":
            holes += 1

        else:
            time_limits += 1

        total_steps += result["steps"]
        total_return += result["discounted_return"]

    return {
        "goals": goals,
        "holes": holes,
        "time_limits": time_limits,
        "mean_steps": total_steps / 50,
        "mean_return": total_return / 50
    }

print("\nORIGINAL MAP RESULTS")

for batch_seed in BATCH_SEEDS:

    result = run_batch(
        ORIGINAL_GRID,
        original_policy,
        batch_seed
    )

    print(
        batch_seed,
        result
    )

print("\nEXTENDED MAP RESULTS")

for batch_seed in BATCH_SEEDS:

    result = run_batch(
        EXTENDED_GRID,
        extended_policy,
        batch_seed
    )

    print(
        batch_seed,
        result
    )
