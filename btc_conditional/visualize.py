import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class RealTimePlotter:
    def __init__(self):
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle("BTC Training Monitor (Conditional LSTM)", fontsize=14, fontweight="bold")
        self.epochs, self.losses, self.scores, self.difficulties, self.times = [], [], [], [], []
        self.loss_line, = self.ax1.plot([], [], "b-", lw=2)
        self.score_line, = self.ax2.plot([], [], "g-", lw=2)
        self.diff_line, = self.ax3.plot([], [], "purple", lw=2)
        self.time_line, = self.ax4.plot([], [], "orange", lw=2)
        self._setup()

    def _setup(self):
        for ax, t, yl in [(self.ax1, "Training Loss", "Loss"), (self.ax2, "Average Score", "Score"),
                           (self.ax3, "Environment Difficulty", "Difficulty"), (self.ax4, "Epoch Time", "Time (s)")]:
            ax.set_title(t, fontsize=11, fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 1); self.ax2.axhline(y=0.7, color="r", ls="--", alpha=0.5)
        self.ax3.set_ylim(0, 1)

    def update(self, epoch, loss, score, diff, t):
        self.epochs.append(epoch); self.losses.append(loss); self.scores.append(score)
        self.difficulties.append(diff); self.times.append(t)
        self.loss_line.set_data(self.epochs, self.losses); self.score_line.set_data(self.epochs, self.scores)
        self.diff_line.set_data(self.epochs, self.difficulties); self.time_line.set_data(self.epochs, self.times)
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]: ax.set_xlim(0, max(self.epochs) + 1)
        if len(self.losses) > 1:
            lo, hi = min(self.losses), max(self.losses); self.ax1.set_ylim(lo - 0.02*abs(lo), hi + 0.02*abs(hi))
        if len(self.times) > 1:
            lo, hi = min(self.times), max(self.times); self.ax4.set_ylim(lo - max(1, 0.1*lo), hi + max(1, 0.1*hi))
        try: self.fig.canvas.draw()
        except: pass

    def close(self):
        try: plt.close()
        except: pass


def plot_final(trainer, out_dir):
    plt.ioff()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("BTC Conditional LSTM — Training Results", fontsize=14, fontweight="bold")
    epochs = range(1, len(trainer.epoch_losses) + 1)
    ax1.plot(epochs, trainer.epoch_losses, "b-", lw=2); ax1.set_title("Training Loss"); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, trainer.epoch_accs, "g-", lw=2); ax2.set_title("Average Score")
    ax2.set_ylim(0, 1); ax2.axhline(y=0.7, color="r", ls="--", alpha=0.5); ax2.grid(True, alpha=0.3)
    ax3.plot(epochs, trainer.epoch_accs, "purple", lw=2); ax3.set_title("Environment Difficulty")
    ax3.set_ylim(0, 1); ax3.grid(True, alpha=0.3)
    ax4.plot(epochs, trainer.epoch_times, "orange", lw=2); ax4.set_title("Epoch Time (s)"); ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "training_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
