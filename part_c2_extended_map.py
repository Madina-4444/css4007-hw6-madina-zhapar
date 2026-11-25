GAMMA = 0.9

GRID = [
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

ROWS = len(GRID)
COLS = len(GRID[0])


def get_tile(row, col):
    return GRID[row - 1][col - 1]


def is_terminal(row, col):
    return get_tile(row, col) in ["G", "H"]


def in_bounds(row, col):
    return 1 <= row <= ROWS and 1 <= col <= COLS


def move(row, col, action):

    dr, dc = ACTIONS[action]

    new_row = row + dr
    new_col = col + dc

    if not in_bounds(new_row, new_col):
        return row, col

    if get_tile(new_row, new_col) == "#":
        return row, col

    return new_row, new_col


def get_reward(row, col):
    return TILE_REWARDS[get_tile(row, col)]


states = []

for row in range(1, ROWS + 1):
    for col in range(1, COLS + 1):

        if get_tile(row, col) != "#":
            states.append((row, col))


print("States:")
print(states)

def get_perpendicular_actions(action):

    if action in ["UP", "DOWN"]:
        return ["LEFT", "RIGHT"]

    return ["UP", "DOWN"]


def get_transitions(row, col, action):

    tile = get_tile(row, col)

    # обычные клетки
    if tile != "I":

        next_row, next_col = move(row, col, action)

        return [
            (1.0, next_row, next_col)
        ]

    # лёд
    transitions = []

    main_row, main_col = move(row, col, action)

    transitions.append(
        (0.8, main_row, main_col)
    )

    for slip_action in get_perpendicular_actions(action):

        slip_row, slip_col = move(
            row,
            col,
            slip_action
        )

        transitions.append(
            (0.1, slip_row, slip_col)
        )

    return transitions


def one_value_iteration(previous_values):

    new_values = {}

    for state in states:

        row, col = state

        if is_terminal(row, col):
            new_values[state] = 0
            continue

        action_values = []

        for action in ACTIONS:

            q = 0

            for prob, next_row, next_col in get_transitions(
                row,
                col,
                action
            ):

                reward = get_reward(
                    next_row,
                    next_col
                )

                next_state = (
                    next_row,
                    next_col
                )

                q += prob * (
                    reward
                    + GAMMA *
                    previous_values[next_state]
                )

            action_values.append(q)

        new_values[state] = max(action_values)

    return new_values

# C2
def value_iteration_until_convergence():

    values = {}

    for state in states:
        values[state] = 0.0

    iteration = 0
    delta = float("inf")

    while delta > 0.000001 and iteration < 10000:

        new_values = one_value_iteration(values)

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

final_values, iterations, final_delta = (
    value_iteration_until_convergence()
)

print("\nCONVERGENCE")
print("Iterations:", iterations)
print("Final delta:", final_delta)

print("\nFINAL VALUES")

for state in sorted(final_values):
    print(
        state,
        round(final_values[state], 4)
    )

# policy extraction
def extract_policy(values):

    policy = {}

    for state in states:

        row, col = state

        if is_terminal(row, col):
            policy[state] = "*"
            continue

        best_action = None
        best_value = float("-inf")

        for action in ACTIONS:

            q = 0

            for prob, next_row, next_col in get_transitions(
                row,
                col,
                action
            ):

                reward = get_reward(
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

ARROWS = {
    "UP": "↑",
    "RIGHT": "→",
    "DOWN": "↓",
    "LEFT": "←",
    "*": "*"
}

policy = extract_policy(final_values)

print("\nPOLICY")

for state in sorted(policy):
    print(state, policy[state])

print("\nPOLICY MAP")

for row in range(1, ROWS + 1):

    line = []

    for col in range(1, COLS + 1):

        if get_tile(row, col) == "#":
            line.append("#")
        else:
            line.append(
                ARROWS[policy[(row, col)]]
            )

    print(" ".join(line))

