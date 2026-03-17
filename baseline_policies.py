from __future__ import annotations

from typing import Callable, Optional

from snake_env import MultiSnakeEnv, Snake, FoodType

PolicyFn = Callable[[MultiSnakeEnv, Snake], int]

AVAILABLE_POLICIES = ("random", "safe_random", "greedy", "heuristic", "model")


def random_policy(env: MultiSnakeEnv, snake: Snake) -> int:
    return env.rng.choice(env.ACTIONS)


def safe_random_policy(env: MultiSnakeEnv, snake: Snake) -> int:
    safe_actions = env._safe_actions(snake)
    if not safe_actions:
        return 0
    return env.rng.choice(safe_actions)


def greedy_food_policy(env: MultiSnakeEnv, snake: Snake) -> int:
    safe_actions = env._safe_actions(snake)
    candidate_actions = safe_actions if safe_actions else list(env.ACTIONS)

    best_action = None
    best_value = -float("inf")
    for action in candidate_actions:
        new_dir = env._rotate_direction(snake.direction, action)
        new_head = env._next_position(snake.head, new_dir)
        value = _position_value(env, snake, new_head)
        if value > best_value:
            best_value = value
            best_action = action
    return int(best_action if best_action is not None else 0)


def heuristic_policy(env: MultiSnakeEnv, snake: Snake) -> int:
    return env._bot_action(snake)


def _position_value(env: MultiSnakeEnv, snake: Snake, new_head: tuple[int, int]) -> float:
    # Préférence pour nourriture riche mais proche; petite pénalité si proche d'une tête adverse.
    best_food_value = -float("inf")
    for pos, food_type in env.foods.items():
        dist = abs(pos[0] - new_head[0]) + abs(pos[1] - new_head[1])
        dist = max(1, dist)
        spec = env.FOOD_SPECS[food_type]
        food_weight = spec.score + 2.0 * spec.reward
        value = food_weight / dist
        if food_type == FoodType.RISKY and snake.length < 4:
            value *= 0.8
        best_food_value = max(best_food_value, value)

    enemy_penalty = 0.0
    nearest_enemy = env._nearest_enemy_head_for_snake(snake.snake_id)
    if nearest_enemy is not None:
        enemy_dist = abs(nearest_enemy[0] - new_head[0]) + abs(nearest_enemy[1] - new_head[1])
        if enemy_dist <= 1:
            enemy_penalty = 4.0
        elif enemy_dist == 2:
            enemy_penalty = 1.5

    wall_margin = min(new_head[0], new_head[1], env.width - 1 - new_head[0], env.height - 1 - new_head[1])
    wall_bonus = 0.3 * max(0, wall_margin)

    return best_food_value + wall_bonus - enemy_penalty


def get_policy_callable(
    policy_name: str,
    model_policy: Optional[PolicyFn] = None,
) -> PolicyFn:
    policy_name = policy_name.strip().lower()
    if policy_name == "random":
        return random_policy
    if policy_name == "safe_random":
        return safe_random_policy
    if policy_name == "greedy":
        return greedy_food_policy
    if policy_name == "heuristic":
        return heuristic_policy
    if policy_name == "model":
        if model_policy is None:
            raise ValueError("Une policy 'model' a été demandée sans modèle chargé.")
        return model_policy
    raise ValueError(f"Policy inconnue: {policy_name}. Possibles: {', '.join(AVAILABLE_POLICIES)}")
