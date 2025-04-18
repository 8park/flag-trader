import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from simulator import MarketEnv
from flag_trader_agent import FlagTraderAgent

def set_seed(seed=42):
    import numpy as np, random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def evaluate(traj):
    pnl_series = [r for (_, _, r) in traj]
    cr = pnl_series[-1] / 10_000 * 100
    sr = pd.Series(pnl_series).mean() / (pd.Series(pnl_series).std() + 1e-8)
    return {"cr": round(cr, 3), "sr": round(sr, 3), "av": 0.0, "mdd": 0.0}

def buy_and_hold(env):
    obs, _ = env.reset()
    env.step(0)  # Buy at t=0
    traj = [(obs, 0, 0)]
    for _ in range(env.t, len(env.df)):
        obs, reward, done, _, _ = env.step(2)  # Hold
        traj.append((obs, 2, reward))
        if done:
            break
    return traj

def plot_results(flag_metrics, bnh_metrics):
    os.makedirs("figures", exist_ok=True)
    time = range(len([flag_metrics["cr"]]))
    plt.figure(figsize=(6, 4))
    plt.plot(time, [flag_metrics["cr"]] * len(time), label="FLAG-TRADER")
    plt.plot(time, [bnh_metrics["cr"]] * len(time), label="Buy-and-Hold")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.savefig("figures/cr_curve.png")
    plt.close()

    df = pd.DataFrame({
        "agent": ["FLAG-TRADER", "Buy-and-Hold"],
        "sr": [flag_metrics["sr"], bnh_metrics["sr"]]
    })
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="agent", y="sr", data=df)
    plt.savefig("figures/sr_boxplot.png")
    plt.close()

def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_seed(cfg["seed"])
    env = MarketEnv("data/MSFT.csv")
    agent = FlagTraderAgent(cfg)
    flag_traj = agent.train_one_episode(env)
    bnh_traj = buy_and_hold(env)
    flag_metrics = evaluate(flag_traj)
    bnh_metrics = evaluate(bnh_traj)
    results = pd.DataFrame([flag_metrics, bnh_metrics], index=["FLAG-TRADER", "Buy-and-Hold"])
    Path("logs").mkdir(exist_ok=True)
    results.to_csv("logs/demo_msft.csv")
    plot_results(flag_metrics, bnh_metrics)
    print("Finished – metrics:", results.to_dict())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arm64.yaml")
    args = ap.parse_args()
    main(args.config)
