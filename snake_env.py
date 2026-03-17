from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np


Position = Tuple[int, int]


class FoodType(str, Enum):
    NORMAL = "normal"
    BONUS = "bonus"
    RISKY = "risky"


@dataclass(frozen=True)
class FoodSpec:
    score: int
    reward: float
    growth: int


@dataclass
class Snake:
    snake_id: int
    body: Deque[Position]
    direction: Position
    is_player: bool = False
    alive: bool = True
    score: int = 0
    growth_pending: int = 0
    last_food: Optional[FoodType] = None

    @property
    def head(self) -> Position:
        return self.body[0]

    @property
    def tail(self) -> Position:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)


class MultiSnakeEnv:
    """
    Environnement Snake multi-agent simplifié, compatible RL.

    API principale:
      - reset() -> observation, info
      - step(action, action_map=None, bot_policy=None) -> observation, reward, terminated, truncated, info
      - render() -> str

    Actions:
      0 = tout droit
      1 = tourner à gauche
      2 = tourner à droite
    """

    ACTIONS = (0, 1, 2)
    DIRECTIONS: Tuple[Position, ...] = (
        (0, -1),   # haut
        (1, 0),    # droite
        (0, 1),    # bas
        (-1, 0),   # gauche
    )

    FOOD_SPECS: Dict[FoodType, FoodSpec] = {
        FoodType.NORMAL: FoodSpec(score=10, reward=10.0, growth=1),
        FoodType.BONUS: FoodSpec(score=18, reward=15.0, growth=1),
        FoodType.RISKY: FoodSpec(score=25, reward=20.0, growth=2),
    }

    FOOD_CHARS = {
        FoodType.NORMAL: "n",
        FoodType.BONUS: "b",
        FoodType.RISKY: "r",
    }

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        n_bots: int = 2,
        max_steps: int = 300,
        seed: Optional[int] = None,
        food_counts: Optional[Dict[FoodType, int]] = None,
        survival_reward: float = 0.2,
        death_penalty: float = -15.0,
        win_bonus: float = 25.0,
        loop_penalty: float = -0.5,
        render_fps: int = 5,
    ) -> None:
        self.width = width
        self.height = height
        self.n_bots = n_bots
        self.max_steps = max_steps
        self.render_fps = render_fps
        self.survival_reward = survival_reward
        self.death_penalty = death_penalty
        self.win_bonus = win_bonus
        self.loop_penalty = loop_penalty

        self.food_counts = food_counts or {
            FoodType.NORMAL: 2,
            FoodType.BONUS: 1,
            FoodType.RISKY: 1,
        }

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.snakes: List[Snake] = []
        self.foods: Dict[Position, FoodType] = {}
        self.steps = 0
        self.player_id = 0
        self.last_player_positions: Deque[Position] = deque(maxlen=8)
        self.action_space_n = 3
        self.observation_space_shape = (self._observation_size(),)

    # ------------------------------------------------------------------
    # API RL
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.steps = 0
        self.foods.clear()
        self.snakes = []
        self.last_player_positions.clear()

        taken: set[Position] = set()
        total_snakes = 1 + self.n_bots

        for snake_id in range(total_snakes):
            snake = self._spawn_snake(snake_id=snake_id, is_player=(snake_id == self.player_id), taken=taken)
            self.snakes.append(snake)
            taken.update(snake.body)

        self._refill_foods()
        if self.player.alive:
            self.last_player_positions.append(self.player.head)
        return self.get_observation(), self._build_info()

    def step(
        self,
        action: int,
        action_map: Optional[Dict[int, int]] = None,
        bot_policy: Optional[Callable[["MultiSnakeEnv", Snake], int]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Action invalide: {action}. Actions attendues: {self.ACTIONS}")

        self.steps += 1
        reward = self.survival_reward
        terminated = False
        truncated = False

        next_dirs: Dict[int, Position] = {}
        next_heads: Dict[int, Position] = {}

        for snake in self.snakes:
            if not snake.alive:
                continue

            if action_map is not None:
                chosen_action = int(action_map.get(snake.snake_id, 0))
            elif snake.is_player:
                chosen_action = action
            else:
                chosen_action = int(bot_policy(self, snake)) if bot_policy is not None else self._bot_action(snake)

            if chosen_action not in self.ACTIONS:
                chosen_action = 0
            next_dir = self._rotate_direction(snake.direction, chosen_action)
            next_head = self._next_position(snake.head, next_dir)
            next_dirs[snake.snake_id] = next_dir
            next_heads[snake.snake_id] = next_head

        deaths: Dict[int, str] = {}

        for snake_id, next_head in next_heads.items():
            if not self._inside(next_head):
                deaths[snake_id] = "wall"

        cell_to_heads: Dict[Position, List[int]] = {}
        for snake_id, next_head in next_heads.items():
            if snake_id in deaths:
                continue
            cell_to_heads.setdefault(next_head, []).append(snake_id)
        for _, snake_ids in cell_to_heads.items():
            if len(snake_ids) > 1:
                for snake_id in snake_ids:
                    deaths[snake_id] = "head_on"

        alive_ids = [s.snake_id for s in self.snakes if s.alive]
        for i, id_a in enumerate(alive_ids):
            for id_b in alive_ids[i + 1 :]:
                if id_a in deaths or id_b in deaths:
                    continue
                snake_a = self.snakes[id_a]
                snake_b = self.snakes[id_b]
                if next_heads.get(id_a) == snake_b.head and next_heads.get(id_b) == snake_a.head:
                    deaths[id_a] = "swap"
                    deaths[id_b] = "swap"

        occupied: set[Position] = set()
        for snake in self.snakes:
            if not snake.alive:
                continue
            body_positions = list(snake.body)
            will_grow = snake.growth_pending > 0
            positions_to_consider = body_positions if will_grow else body_positions[:-1]
            occupied.update(positions_to_consider)

        for snake_id, next_head in next_heads.items():
            if snake_id in deaths:
                continue
            if next_head in occupied:
                deaths[snake_id] = "body"

        bots_dead_now = 0
        for snake_id, _reason in deaths.items():
            snake = self.snakes[snake_id]
            snake.alive = False
            if snake.is_player:
                reward += self.death_penalty
                terminated = True
            else:
                bots_dead_now += 1

        for snake in self.snakes:
            if not snake.alive:
                continue
            snake.direction = next_dirs[snake.snake_id]
            snake.body.appendleft(next_heads[snake.snake_id])
            snake.last_food = None

            ate_food = next_heads[snake.snake_id] in self.foods
            if ate_food:
                food_type = self.foods.pop(next_heads[snake.snake_id])
                spec = self.FOOD_SPECS[food_type]
                snake.score += spec.score
                snake.growth_pending += spec.growth
                snake.last_food = food_type
                if snake.is_player:
                    reward += spec.reward
            if snake.growth_pending > 0:
                snake.growth_pending -= 1
            else:
                snake.body.pop()

        self._refill_foods()

        if self.player.alive:
            self.last_player_positions.append(self.player.head)
            if len(self.last_player_positions) == self.last_player_positions.maxlen:
                unique_positions = len(set(self.last_player_positions))
                if unique_positions <= 3:
                    reward += self.loop_penalty

        alive_non_player = sum(1 for s in self.snakes if s.alive and not s.is_player)
        if self.player.alive and bots_dead_now > 0:
            reward += 0.5 * bots_dead_now
        if self.player.alive and alive_non_player == 0:
            reward += self.win_bonus
            terminated = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            if self.player.alive and self._is_player_first():
                reward += self.win_bonus / 2.0

        info = self._build_info(deaths=deaths, terminated=terminated, truncated=truncated)
        return self.get_observation(), reward, terminated, truncated, info

    def get_observation(self) -> np.ndarray:
        return self.get_observation_for_snake(self.player_id)

    def get_observation_for_snake(self, snake_id: int) -> np.ndarray:
        return self._observation_for_snake(snake_id)

    def render(self) -> str:
        board = [["." for _ in range(self.width)] for _ in range(self.height)]

        for pos, food_type in self.foods.items():
            x, y = pos
            board[y][x] = self.FOOD_CHARS[food_type]

        for snake in self.snakes:
            char_head = "P" if snake.is_player else str(snake.snake_id)
            char_body = "p" if snake.is_player else str(snake.snake_id).lower()
            for idx, (x, y) in enumerate(snake.body):
                if not snake.alive:
                    board[y][x] = "x"
                else:
                    board[y][x] = char_head if idx == 0 else char_body

        lines = ["+" + "-" * self.width + "+"]
        for row in board:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.width + "+")
        lines.append(f"step={self.steps} | player_alive={self.player.alive} | score={self.player.score} | length={self.player.length}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fonctionnement interne
    # ------------------------------------------------------------------
    @property
    def player(self) -> Snake:
        return self.snakes[self.player_id]

    def _observation_size(self) -> int:
        return 23

    def _observation_for_snake(self, snake_id: int) -> np.ndarray:
        snake = self.snakes[snake_id]
        if not snake.alive:
            return np.zeros(self._observation_size(), dtype=np.float32)

        head = snake.head
        direction = snake.direction
        left_dir = self._rotate_direction(direction, 1)
        right_dir = self._rotate_direction(direction, 2)

        danger_straight = float(self._is_danger_for_snake(snake_id, head, direction))
        danger_left = float(self._is_danger_for_snake(snake_id, head, left_dir))
        danger_right = float(self._is_danger_for_snake(snake_id, head, right_dir))

        dir_one_hot = [1.0 if direction == d else 0.0 for d in self.DIRECTIONS]

        food_features: List[float] = []
        for food_type in (FoodType.NORMAL, FoodType.BONUS, FoodType.RISKY):
            pos = self._nearest_food(head, food_type)
            if pos is None:
                food_features.extend([0.0, 0.0])
            else:
                dx = (pos[0] - head[0]) / max(1, self.width - 1)
                dy = (pos[1] - head[1]) / max(1, self.height - 1)
                food_features.extend([dx, dy])

        x, y = head
        wall_features = [
            y / max(1, self.height - 1),
            (self.width - 1 - x) / max(1, self.width - 1),
            (self.height - 1 - y) / max(1, self.height - 1),
            x / max(1, self.width - 1),
        ]

        nearest_enemy = self._nearest_enemy_head(head, observer_id=snake_id)
        if nearest_enemy is None:
            enemy_features = [0.0, 0.0, 0.0]
        else:
            dx = (nearest_enemy[0] - head[0]) / max(1, self.width - 1)
            dy = (nearest_enemy[1] - head[1]) / max(1, self.height - 1)
            dist = (abs(nearest_enemy[0] - head[0]) + abs(nearest_enemy[1] - head[1])) / max(1, self.width + self.height)
            enemy_features = [dx, dy, dist]

        misc_features = [
            snake.length / max(1, self.width * self.height),
            snake.score / 100.0,
            sum(1 for s in self.snakes if s.alive) / max(1, len(self.snakes)),
        ]

        return np.array(
            [danger_straight, danger_left, danger_right] + dir_one_hot + food_features + wall_features + enemy_features + misc_features,
            dtype=np.float32,
        )

    def _spawn_snake(self, snake_id: int, is_player: bool, taken: set[Position]) -> Snake:
        while True:
            direction = self.rng.choice(self.DIRECTIONS)
            head = (self.rng.randint(2, self.width - 3), self.rng.randint(2, self.height - 3))
            body = deque([head])
            valid = True
            current = head
            for _ in range(2):
                current = (current[0] - direction[0], current[1] - direction[1])
                if not self._inside(current) or current in taken:
                    valid = False
                    break
                body.append(current)
            if not valid:
                continue
            if any(pos in taken for pos in body):
                continue
            return Snake(snake_id=snake_id, body=body, direction=direction, is_player=is_player)

    def _build_info(self, deaths: Optional[Dict[int, str]] = None, terminated: bool = False, truncated: bool = False) -> Dict:
        rankings = sorted(
            [
                {
                    "snake_id": s.snake_id,
                    "is_player": s.is_player,
                    "alive": s.alive,
                    "score": s.score,
                    "length": s.length,
                }
                for s in self.snakes
            ],
            key=lambda x: (x["score"], x["alive"], x["length"]),
            reverse=True,
        )
        player_rank = next((i + 1 for i, item in enumerate(rankings) if item["snake_id"] == self.player_id), len(rankings))

        outcome = "ongoing"
        if terminated and self.player.alive:
            outcome = "win_elimination"
        elif terminated and not self.player.alive:
            outcome = "loss_death"
        elif truncated and self.player.alive and player_rank == 1:
            outcome = "win_timeout"
        elif truncated and self.player.alive and len(rankings) > 1 and rankings[0]["score"] == self.player.score:
            outcome = "draw_timeout"
        elif truncated:
            outcome = "loss_timeout"

        return {
            "steps": self.steps,
            "deaths": deaths or {},
            "player_alive": self.player.alive,
            "player_score": self.player.score,
            "player_length": self.player.length,
            "player_rank": player_rank,
            "rankings": rankings,
            "outcome": outcome,
            "is_win": outcome.startswith("win_"),
            "is_draw": outcome.startswith("draw_"),
            "is_loss": outcome.startswith("loss_"),
            "foods": {f"{x},{y}": ft.value for (x, y), ft in self.foods.items()},
        }

    def _inside(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _rotate_direction(self, current_dir: Position, action: int) -> Position:
        idx = self.DIRECTIONS.index(current_dir)
        if action == 0:
            return current_dir
        if action == 1:
            return self.DIRECTIONS[(idx - 1) % 4]
        return self.DIRECTIONS[(idx + 1) % 4]

    def _next_position(self, head: Position, direction: Position) -> Position:
        return (head[0] + direction[0], head[1] + direction[1])

    def _occupied_positions(self) -> set[Position]:
        occupied = set()
        for snake in self.snakes:
            if snake.alive:
                occupied.update(snake.body)
        return occupied

    def _refill_foods(self) -> None:
        for food_type, wanted in self.food_counts.items():
            current = sum(1 for ft in self.foods.values() if ft == food_type)
            while current < wanted:
                pos = self._random_empty_cell()
                self.foods[pos] = food_type
                current += 1

    def _random_empty_cell(self) -> Position:
        occupied = self._occupied_positions().union(self.foods.keys())
        free_cells = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        if not free_cells:
            raise RuntimeError("Aucune case libre disponible pour placer un élément.")
        return self.rng.choice(free_cells)

    def _nearest_food(self, head: Position, food_type: FoodType) -> Optional[Position]:
        candidates = [pos for pos, ft in self.foods.items() if ft == food_type]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))

    def _nearest_enemy_head(self, head: Position, observer_id: Optional[int] = None) -> Optional[Position]:
        candidates = [s.head for s in self.snakes if s.alive and (observer_id is None or s.snake_id != observer_id)]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))

    def _is_player_first(self) -> bool:
        player_tuple = (self.player.score, self.player.alive, self.player.length)
        best_tuple = max((s.score, s.alive, s.length) for s in self.snakes)
        return player_tuple == best_tuple

    def _is_danger(self, head: Position, direction: Position, ignore_player_tail: bool = True) -> bool:
        next_pos = self._next_position(head, direction)
        if not self._inside(next_pos):
            return True

        occupied = set()
        for snake in self.snakes:
            if not snake.alive:
                continue
            positions = list(snake.body)
            if snake.is_player and ignore_player_tail and snake.growth_pending == 0:
                positions = positions[:-1]
            occupied.update(positions)
        return next_pos in occupied

    def _is_danger_for_snake(self, snake_id: int, head: Position, direction: Position) -> bool:
        next_pos = self._next_position(head, direction)
        if not self._inside(next_pos):
            return True

        occupied = set()
        for snake in self.snakes:
            if not snake.alive:
                continue
            positions = list(snake.body)
            if snake.snake_id == snake_id and snake.growth_pending == 0:
                positions = positions[:-1]
            occupied.update(positions)
        return next_pos in occupied

    def _safe_actions(self, snake: Snake) -> List[int]:
        safe = []
        for action in self.ACTIONS:
            new_dir = self._rotate_direction(snake.direction, action)
            new_head = self._next_position(snake.head, new_dir)
            if not self._inside(new_head):
                continue

            collision = False
            for other in self.snakes:
                if not other.alive:
                    continue
                positions = list(other.body)
                if other.snake_id == snake.snake_id and snake.growth_pending == 0:
                    positions = positions[:-1]
                if new_head in positions:
                    collision = True
                    break
            if not collision:
                safe.append(action)
        return safe

    def _bot_action(self, snake: Snake) -> int:
        safe_actions = self._safe_actions(snake)
        if not safe_actions:
            return 0

        target = self._nearest_food(snake.head, FoodType.BONUS)
        if target is None:
            target = self._nearest_food(snake.head, FoodType.NORMAL)
        if target is None:
            target = self._nearest_food(snake.head, FoodType.RISKY)
        if target is None:
            return self.rng.choice(safe_actions)

        best_action = None
        best_distance = None
        for action in safe_actions:
            new_dir = self._rotate_direction(snake.direction, action)
            new_head = self._next_position(snake.head, new_dir)
            dist = abs(new_head[0] - target[0]) + abs(new_head[1] - target[1])
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_action = action
        return 0 if best_action is None else best_action
