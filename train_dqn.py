from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import DQNAgent
from snake_env import MultiSnakeEnv


def moving_average(values: List[float], window: int = 50) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def build_env(args: argparse.Namespace, n_bots: int | None = None) -> MultiSnakeEnv:
    return MultiSnakeEnv(
        width=args.width,
        height=args.height,
        n_bots=args.n_bots if n_bots is None else n_bots,
        max_steps=args.max_steps,
        seed=args.seed,
        survival_reward=args.survival_reward,
        death_penalty=args.death_penalty,
        win_bonus=args.win_bonus,
        timeout_win_bonus=args.timeout_win_bonus,
        draw_bonus=args.draw_bonus,
        timeout_loss_penalty=args.timeout_loss_penalty,
        elimination_bonus=args.elimination_bonus,
        loop_penalty=args.loop_penalty,
    )


def evaluate_agent(agent: DQNAgent, args: argparse.Namespace, episodes: int, n_bots: int | None = None) -> Dict[str, float]:
    env = build_env(args, n_bots=n_bots)
    rewards: List[float] = []
    scores: List[float] = []
    wins: List[float] = []
    draws: List[float] = []
    ranks: List[float] = []

    for episode in range(episodes):
        state, _ = env.reset(seed=args.seed + 10_000 + episode)
        done = False
        total_reward = 0.0
        last_info = {}
        while not done:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            last_info = info
        rewards.append(total_reward)
        scores.append(float(last_info.get("player_score", 0.0)))
        wins.append(1.0 if last_info.get("is_win", False) else 0.0)
        draws.append(1.0 if last_info.get("is_draw", False) else 0.0)
        ranks.append(float(last_info.get("player_rank", env.n_bots + 1)))

    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "win_rate": float(np.mean(wins)),
        "draw_rate": float(np.mean(draws)),
        "avg_rank": float(np.mean(ranks)),
    }


def train(args: argparse.Namespace) -> Dict[str, List[float]]:
    env = build_env(args, n_bots=1 if args.curriculum else None)
    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        learning_starts=args.learning_starts,
        double_dqn=not args.disable_double_dqn,
        device=args.device,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards: List[float] = []
    episode_scores: List[float] = []
    episode_lengths: List[float] = []
    episode_wins: List[float] = []
    episode_draws: List[float] = []
    episode_ranks: List[float] = []
    losses: List[float] = []
    eval_rewards: List[float] = []
    eval_scores: List[float] = []
    eval_win_rates: List[float] = []
    eval_draw_rates: List[float] = []
    eval_avg_ranks: List[float] = []
    eval_checkpoints: List[int] = []

    best_eval_score = -float("inf")

    for episode in range(1, args.episodes + 1):
        if args.curriculum:
            current_bots = 1 if episode <= args.curriculum_stage_1 else (2 if episode <= args.curriculum_stage_2 else args.n_bots)
            env = build_env(args, n_bots=current_bots)
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1
            last_info = info

        agent.decay_epsilon()

        player_score = float(last_info.get("player_score", 0.0))
        player_alive = bool(last_info.get("player_alive", False))
        is_win = 1.0 if last_info.get("is_win", False) else 0.0
        is_draw = 1.0 if last_info.get("is_draw", False) else 0.0
        player_rank = float(last_info.get("player_rank", env.n_bots + 1))

        episode_rewards.append(total_reward)
        episode_scores.append(player_score)
        episode_lengths.append(float(steps))
        episode_wins.append(is_win)
        episode_draws.append(is_draw)
        episode_ranks.append(player_rank)

        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_metrics = evaluate_agent(agent, args, episodes=args.eval_episodes)
            eval_rewards.append(eval_metrics["reward"])
            eval_scores.append(eval_metrics["score"])
            eval_win_rates.append(eval_metrics["win_rate"])
            eval_draw_rates.append(eval_metrics["draw_rate"])
            eval_avg_ranks.append(eval_metrics["avg_rank"])
            eval_checkpoints.append(episode)
            combined_eval_score = (
                120.0 * eval_metrics["win_rate"]
                + 20.0 * eval_metrics["draw_rate"]
                + 0.35 * eval_metrics["score"]
                + 0.10 * eval_metrics["reward"]
                - 2.0 * eval_metrics["avg_rank"]
            )
            if combined_eval_score > best_eval_score:
                best_eval_score = combined_eval_score
                agent.save(models_dir / "best_eval_model.pt")

        if episode % args.log_every == 0 or episode == 1 or episode == args.episodes:
            avg_window = min(args.log_every, len(episode_rewards))
            eval_suffix = ""
            if eval_rewards:
                eval_suffix = (
                    f" | eval_reward={eval_rewards[-1]:7.2f}"
                    f" | eval_score={eval_scores[-1]:5.2f}"
                    f" | eval_win={eval_win_rates[-1]:.2%}"
                    f" | eval_draw={eval_draw_rates[-1]:.2%}"
                    f" | eval_rank={eval_avg_ranks[-1]:.2f}"
                )
            print(
                f"Episode {episode:4d}/{args.episodes} | "
                f"reward={total_reward:7.2f} | score={player_score:5.1f} | "
                f"steps={steps:4d} | alive={player_alive} | "
                f"eps={agent.epsilon:.3f} | "
                f"avg_reward_{avg_window}={mean(episode_rewards[-avg_window:]):7.2f} | "
                f"avg_score_{avg_window}={mean(episode_scores[-avg_window:]):5.2f}"
                + eval_suffix
            )

    agent.save(models_dir / "last_model.pt")

    history = {
        "episode_rewards": episode_rewards,
        "episode_scores": episode_scores,
        "episode_lengths": episode_lengths,
        "episode_wins": episode_wins,
        "episode_draws": episode_draws,
        "episode_ranks": episode_ranks,
        "losses": losses,
        "reward_ma": moving_average(episode_rewards, window=args.ma_window),
        "score_ma": moving_average(episode_scores, window=args.ma_window),
        "win_rate_ma": moving_average(episode_wins, window=args.ma_window),
        "draw_rate_ma": moving_average(episode_draws, window=args.ma_window),
        "rank_ma": moving_average(episode_ranks, window=args.ma_window),
        "eval_checkpoints": eval_checkpoints,
        "eval_rewards": eval_rewards,
        "eval_scores": eval_scores,
        "eval_win_rates": eval_win_rates,
        "eval_draw_rates": eval_draw_rates,
        "eval_avg_ranks": eval_avg_ranks,
        "config": vars(args),
    }

    with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, plots_dir)
    return history


def plot_training_curves(history: Dict[str, List[float]], plots_dir: Path) -> None:
    rewards = history["episode_rewards"]
    reward_ma = history["reward_ma"]
    scores = history["episode_scores"]
    score_ma = history["score_ma"]
    wins_ma = history["win_rate_ma"]
    eval_x = history.get("eval_checkpoints", [])
    eval_rewards = history.get("eval_rewards", [])
    eval_scores = history.get("eval_scores", [])
    eval_wins = history.get("eval_win_rates", [])
    eval_ranks = history.get("eval_avg_ranks", [])
    rank_ma = history.get("rank_ma", [])

    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.4, label="Reward train")
    plt.plot(reward_ma, linewidth=2, label="Reward moyenne glissante")
    if eval_x:
        plt.plot(eval_x, eval_rewards, marker="o", linestyle="--", label="Reward eval")
    plt.xlabel("Épisodes")
    plt.ylabel("Reward")
    plt.title("Évolution de la reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "rewards.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(scores, alpha=0.4, label="Score train")
    plt.plot(score_ma, linewidth=2, label="Score moyen glissant")
    if eval_x:
        plt.plot(eval_x, eval_scores, marker="o", linestyle="--", label="Score eval")
    plt.xlabel("Épisodes")
    plt.ylabel("Score")
    plt.title("Évolution du score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "scores.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(wins_ma, linewidth=2, label="Win rate train glissant")
    if eval_x:
        plt.plot(eval_x, eval_wins, marker="o", linestyle="--", label="Win rate eval")
    plt.xlabel("Épisodes")
    plt.ylabel("Win rate")
    plt.title("Taux de victoire")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "win_rate.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(rank_ma, linewidth=2, label="Rang moyen train glissant")
    if eval_x:
        plt.plot(eval_x, eval_ranks, marker="o", linestyle="--", label="Rang moyen eval")
    plt.xlabel("Épisodes")
    plt.ylabel("Rang")
    plt.title("Rang moyen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "rank.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Double Dueling DQN pour Multi-Snake")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--n-bots", dest="n_bots", type=int, default=2)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=300)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--buffer-capacity", dest="buffer_capacity", type=int, default=100000)
    parser.add_argument("--target-update-freq", dest="target_update_freq", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--learning-starts", dest="learning_starts", type=int, default=1000)
    parser.add_argument("--epsilon-start", dest="epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", dest="epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", dest="epsilon_decay", type=float, default=0.998)
    parser.add_argument("--disable-double-dqn", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-stage-1", dest="curriculum_stage_1", type=int, default=500)
    parser.add_argument("--curriculum-stage-2", dest="curriculum_stage_2", type=int, default=1000)
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=100)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=20)
    parser.add_argument("--ma-window", dest="ma_window", type=int, default=50)
    parser.add_argument("--log-every", dest="log_every", type=int, default=50)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="outputs_v3")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--survival-reward", dest="survival_reward", type=float, default=0.03)
    parser.add_argument("--death-penalty", dest="death_penalty", type=float, default=-20.0)
    parser.add_argument("--win-bonus", dest="win_bonus", type=float, default=35.0)
    parser.add_argument("--timeout-win-bonus", dest="timeout_win_bonus", type=float, default=18.0)
    parser.add_argument("--draw-bonus", dest="draw_bonus", type=float, default=4.0)
    parser.add_argument("--timeout-loss-penalty", dest="timeout_loss_penalty", type=float, default=-3.0)
    parser.add_argument("--elimination-bonus", dest="elimination_bonus", type=float, default=1.5)
    parser.add_argument("--loop-penalty", dest="loop_penalty", type=float, default=-1.5)

    args = parser.parse_args()
    history = train(args)

    print("\nEntraînement terminé.")
    print(f"Dernière reward moyenne glissante: {history['reward_ma'][-1]:.2f}")
    print(f"Dernier score moyen glissant: {history['score_ma'][-1]:.2f}")
    print(f"Dernier win rate glissant: {history['win_rate_ma'][-1]:.2f}")
    print(f"Dernier draw rate glissant: {history['draw_rate_ma'][-1]:.2f}")
    print(f"Dernier rang moyen glissant: {history['rank_ma'][-1]:.2f}")
    if history["eval_rewards"]:
        print(f"Dernière reward eval: {history['eval_rewards'][-1]:.2f}")
        print(f"Dernier score eval: {history['eval_scores'][-1]:.2f}")
        print(f"Dernier win rate eval: {history['eval_win_rates'][-1]:.2%}")
        print(f"Dernier draw rate eval: {history['eval_draw_rates'][-1]:.2%}")
        print(f"Dernier rang moyen eval: {history['eval_avg_ranks'][-1]:.2f}")
