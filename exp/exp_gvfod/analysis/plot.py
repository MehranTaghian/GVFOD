import os
import click
from matplotlib import pyplot as plt
import matplotlib.collections
import numpy as np
import pandas as pd
import seaborn as sns


@click.command()
@click.argument("src", nargs=1)
@click.argument("dst", nargs=1)
class Plot:
    def __init__(self, src, dst):
        plt.rcParams["font.family"] = "Times New Roman"
        self.data = pd.read_csv(src)
        self.dst = dst
        for c in self.data.columns.tolist():
            if 'precision' in c or 'recall' in c or 'f1_score' in c:
                self.plot_avg_performance_heatmap(c)
                self.plot_var_performance_stacked(c)

    def plot_avg_performance_heatmap(self, metric_fault):
        avg = self.data[['Algorithm', 'Training Size', metric_fault]]
        avg = avg.groupby(['Algorithm', 'Training Size']).agg({metric_fault: 'mean'}).reset_index()
        avg = pd.pivot(avg, index='Algorithm', columns='Training Size', values=metric_fault)
        plt.figure(figsize=(12, 5))
        sns.heatmap(avg)
        plt.savefig(os.path.join(self.dst, f'heatmap_{metric_fault}.jpg'), dpi=300)
        plt.close()

    def plot_var_performance_stacked(self, metric_fault):
        var = self.data[['Algorithm', 'Training Size', metric_fault]]
        var = var.groupby(['Algorithm', 'Training Size']).agg({metric_fault: 'var'}).reset_index()
        # Decide Colors
        mycolors = sns.color_palette('tab20', 11)

        # Draw Plot and Annotate
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)

        algorithms = pd.unique(var['Algorithm']).tolist()
        x = pd.unique(var['Training Size']).tolist()
        y = np.zeros([len(algorithms), len(x)])

        for i, a in enumerate(algorithms):
            y[i] = np.array(var[var['Algorithm'] == a][metric_fault])

        # Plot for each column
        ax = plt.gca()
        ax.stackplot(x, y, labels=algorithms, colors=mycolors, alpha=0.8)

        # Decorations
        ax.set_title('Night Visitors in Australian Regions', fontsize=18)
        ax.set(ylim=[0, 1])
        ax.legend(fontsize=10, ncol=4)
        plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
        plt.yticks(np.arange(0, 1, 0.1), fontsize=10)
        plt.xlim(x[0], x[-1])

        # Lighten borders
        plt.gca().spines["top"].set_alpha(0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(0)
        plt.gca().spines["left"].set_alpha(.3)
        plt.savefig(os.path.join(self.dst, f'stacked_{metric_fault}.jpg'), dpi=300)

if __name__ == "__main__":
    p = Plot()
