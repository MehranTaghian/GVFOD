import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


def convert_var_to_count(var, metric_fault):
    algorithms, training_size = pd.unique(var['Algorithm']), pd.unique(var['Training Size'])

    var[metric_fault] = (var[metric_fault] - var[metric_fault].min()) / (
            var[metric_fault].max() - var[metric_fault].min())

    for a in algorithms:
        for t in training_size:
            variance = var.loc[(var['Algorithm'] == a) & (var['Training Size'] == t)][metric_fault].iloc[0]
            for _ in range(int(100 * variance)):
                new_row = {'Algorithm': a, 'Training Size': t, metric_fault: 0}
                var = var.append(new_row, ignore_index=True)

    return var


class Plot:
    def __init__(self, src, dst):
        # plt.rcParams["font.family"] = "fantasy"
        mpl.rcParams['font.family'] = ['serif']
        mpl.rcParams['font.serif'] = ['Times New Roman']
        params = {'axes.titlesize': 20, 'axes.labelsize': 15, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
        mpl.rcParams.update(params)

        self.axis_fontsize = 15
        self.title_fontsize = 20
        self.data = pd.read_csv(src)
        self.dst = dst
        fig, ax = plt.subplots(3, 2, figsize=[24, 12], sharex='col')
        row = 0
        for c in self.data.columns.tolist():
            if 'precision' in c or 'recall' in c or 'f1_score' in c:
                self.plot_avg_performance_heatmap(c, ax[row, 0])
                self.plot_var_performance_violin(c, ax[row, 1])
                # self.plot_var_performance_stacked(c, ax[row, 1])
                row += 1

        # ax[0, 1].legend(fontsize=10, ncol=4)
        ax[-1, 0].set_xlabel('Training Size', fontsize=self.axis_fontsize)
        ax[-1, 1].set_xlabel('Algorithms', fontsize=self.axis_fontsize)
        training_size = pd.unique(self.data['Training Size']).tolist()
        labels = ['' for _ in range(len(training_size))]
        labels[0], labels[int(len(labels) / 2)], labels[-1] = 50, 1050, 1950
        ax[-1, 0].set_xticklabels(labels)
        filename = (os.path.split(src)[1]) \
            .replace(".csv", ".png") \
            .replace("train_size_delay_0_", "3-")
        fig.tight_layout()
        fig.savefig(os.path.join(dst, filename), dpi=300)

    def plot_avg_performance_heatmap(self, metric_fault, ax):
        avg = self.data[['Algorithm', 'Training Size', metric_fault]]
        avg = avg.groupby(['Algorithm', 'Training Size']).agg({metric_fault: 'mean'}).reset_index()
        avg = pd.pivot(avg, index='Algorithm', columns='Training Size', values=metric_fault)
        sns.heatmap(avg, ax=ax)
        metric = ' '.join(metric_fault.split('_')[:2])
        ax.set_title(f'Average {metric}')
        ax.set_ylabel('Algorithm')
        ax.set_xlabel('')

    def plot_var_performance_violin(self, metric_fault, ax):
        var = self.data[['Algorithm', 'Training Size', metric_fault]]
        var = var.groupby(['Algorithm', 'Training Size']).agg({metric_fault: 'var'}).reset_index()
        var = convert_var_to_count(var, metric_fault)
        sns.violinplot(data=var, x="Algorithm", y="Training Size", scale="count", ax=ax, cut=0)
        ax.set_xlabel('')
        ax.set_ylabel('Training Size')
        ax.set_title(f"Variance of {' '.join(metric_fault.split('_')[:2])}")

    def plot_var_performance_stacked(self, metric_fault, ax):
        var = self.data[['Algorithm', 'Training Size', metric_fault]]
        var = var.groupby(['Algorithm', 'Training Size']).agg({metric_fault: 'var'}).reset_index()
        # Decide Colors
        mycolors = sns.color_palette('tab20', 11)

        algorithms = pd.unique(var['Algorithm']).tolist()
        x = pd.unique(var['Training Size']).tolist()
        y = np.zeros([len(algorithms), len(x)])

        for i, a in enumerate(algorithms):
            y[i] = np.array(var[var['Algorithm'] == a][metric_fault])

        # Plot for each column
        ax.stackplot(x, y, labels=algorithms, colors=mycolors, alpha=0.8)

        # Decorations
        ax.set_yticks(np.arange(0, 1, 0.1))  # , fontsize=10)
        ax.set_xlim(x[0], x[-1])
        metric = ' '.join(metric_fault.split('_')[:2])
        ax.set(ylim=[0, 1])
        ax.set_ylabel(f'Variance of {metric}')

        # Lighten borders
        ax.spines["top"].set_alpha(0)
        ax.spines["bottom"].set_alpha(.3)
        ax.spines["right"].set_alpha(0)
        ax.spines["left"].set_alpha(.3)


if __name__ == "__main__":
    src = Path(os.path.abspath(__file__)).parent.parent
    csv = os.path.join(src, 'csv')
    dst = os.path.join(src, 'figures')
    if not os.path.exists(dst):
        os.makedirs(dst)
    for csv_path in os.listdir(csv):
        p = Plot(os.path.join(csv, csv_path), dst)
