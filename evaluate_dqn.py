from __future__ import annotations

import argparse
from statistics import mean

from dqn_agent import DQNAgent
from snake_env import MultiSnakeEnv


def evaluate(args: argparse.Namespace) -> None:
    env = MultiSnakeEnv(
        width=args.width,
        height=args.height,
        n_bots=args.n_bots,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed,
    )
    agent.load(args.model_path)

    scores = []
    rewards = []
    wins = []
    draws = []
    ranks = []

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + 1000 + episode)
        done = False
        total_reward = 0.0
        final_info = {}

        while not done:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            final_info = info
            if args.render:
                print(env.render())
                print(f"reward={reward:.2f} | outcome={info['outcome']}")
                print("-" * 50)

        rewards.append(total_reward)
        scores.append(float(final_info.get("player_score", 0.0)))
        wins.append(1.0 if final_info.get("is_win", False) else 0.0)
        draws.append(1.0 if final_info.get("is_draw", False) else 0.0)
        ranks.append(float(final_info.get("player_rank", env.n_bots + 1)))

    print(f"Épisodes évalués : {args.episodes}")
    print(f"Reward moyenne    : {mean(rewards):.2f}")
    print(f"Score moyen       : {mean(scores):.2f}")
    print(f"Taux de victoire  : {mean(wins):.2%}")
    print(f"Taux de draw      : {mean(draws):.2%}")
    print(f"Rang moyen        : {mean(ranks):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle DQN Multi-Snake")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--n-bots", dest="n_bots", type=int, default=2)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=300)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    evaluate(parser.parse_args())
