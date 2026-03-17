from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional

from baseline_policies import AVAILABLE_POLICIES, get_policy_callable
from dqn_agent import DQNAgent
from snake_env import MultiSnakeEnv, Snake

PolicyFn = Callable[[MultiSnakeEnv, Snake], int]


def build_env(args: argparse.Namespace) -> MultiSnakeEnv:
    return MultiSnakeEnv(
        width=args.width,
        height=args.height,
        n_bots=args.n_bots,
        max_steps=args.max_steps,
        seed=args.seed,
    )


def load_model_policy(args: argparse.Namespace, env: MultiSnakeEnv) -> Optional[PolicyFn]:
    if not args.model_path:
        return None
    agent = DQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed,
    )
    agent.load(args.model_path)

    def model_policy(local_env: MultiSnakeEnv, snake: Snake) -> int:
        state = local_env.get_observation_for_snake(snake.snake_id)
        return int(agent.act(state, greedy=True))

    return model_policy


def evaluate_policy(
    policy_name: str,
    args: argparse.Namespace,
    model_policy: Optional[PolicyFn],
) -> Dict[str, float | str]:
    env = build_env(args)
    player_policy = get_policy_callable(policy_name, model_policy=model_policy)
    bot_policy = get_policy_callable(args.bot_policy, model_policy=model_policy)

    rewards: List[float] = []
    scores: List[float] = []
    wins: List[float] = []
    draws: List[float] = []
    ranks: List[float] = []
    steps_list: List[float] = []

    for episode in range(args.episodes):
        env.reset(seed=args.seed + 5000 + episode)
        done = False
        total_reward = 0.0
        last_info: Dict = {}

        while not done:
            action = player_policy(env, env.player)
            _, reward, terminated, truncated, info = env.step(action, bot_policy=bot_policy)
            total_reward += reward
            done = terminated or truncated
            last_info = info

        rewards.append(total_reward)
        scores.append(float(last_info.get("player_score", 0.0)))
        wins.append(1.0 if last_info.get("is_win", False) else 0.0)
        draws.append(1.0 if last_info.get("is_draw", False) else 0.0)
        ranks.append(float(last_info.get("player_rank", env.n_bots + 1)))
        steps_list.append(float(last_info.get("steps", 0)))

    return {
        "policy": policy_name,
        "reward": round(mean(rewards), 2),
        "score": round(mean(scores), 2),
        "win_rate": round(mean(wins), 4),
        "draw_rate": round(mean(draws), 4),
        "avg_rank": round(mean(ranks), 2),
        "avg_steps": round(mean(steps_list), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark des baselines et du modèle DQN sur Multi-Snake")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--n-bots", dest="n_bots", type=int, default=2)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--bot-policy", type=str, default="heuristic", choices=AVAILABLE_POLICIES)
    parser.add_argument(
        "--policies",
        type=str,
        nargs="*",
        default=["random", "safe_random", "greedy", "heuristic", "model"],
        help="Policies joueur à comparer",
    )
    parser.add_argument("--output-json", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    env = build_env(args)
    model_policy = load_model_policy(args, env)

    results: List[Dict[str, float | str]] = []
    for policy_name in args.policies:
        if policy_name == "model" and model_policy is None:
            continue
        result = evaluate_policy(policy_name, args, model_policy)
        results.append(result)

    results.sort(key=lambda x: (x["win_rate"], -x["avg_rank"], x["score"]), reverse=True)

    print(f"Benchmark sur {args.episodes} épisodes | bots={args.bot_policy}")
    print("policy       | reward | score | win%   | draw%  | rank | steps")
    print("-" * 66)
    for row in results:
        print(
            f"{row['policy']:<12} | {row['reward']:>6.2f} | {row['score']:>5.2f} | "
            f"{100*row['win_rate']:>5.1f}% | {100*row['draw_rate']:>5.1f}% | "
            f"{row['avg_rank']:>4.2f} | {row['avg_steps']:>5.1f}"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés dans: {output_path}")


if __name__ == "__main__":
    main()
