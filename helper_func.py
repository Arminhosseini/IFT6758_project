import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve

def Visualize_ROC(models, probs, y_true, path_output_image):
    fig = plt.figure()

    for i in range(len(models)):
        fpr, tpr, _ = roc_curve(y_true, probs[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{models[i]} (area = {round(roc_auc, 2)})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(path_output_image)


def Visualize_Goal_Rate(models, probs, y_true, path_output_image):
    bins = np.linspace(0, 95, num=20).astype(int)
    fig, axs = plt.subplots(1, 1)

    list_plot_points = list()
    for i in range(len(models)):
        perentiles = (np.argsort(np.argsort(probs[i])) + 1) * 100.0 / (len(probs[i]))
        perentile_bins = (np.digitize(perentiles, bins) - 1) * 5
        plot_points = pd.DataFrame(columns=["bin", "goal_rate", "goal_cum"])
        goal_cum = 0
        for j, bin in enumerate(np.flip(bins)):
            n_goal = np.sum(y_true[perentile_bins == bin])
            n_shot = np.sum(perentile_bins == bin)
            goal_rate = n_goal / n_shot
            goal_cum += n_goal / np.sum(y_true)
            plot_points.loc[j] = [bin, goal_rate, goal_cum]
        list_plot_points.append(plot_points)

        sns.lineplot(
            data=plot_points,
            x="bin",
            y="goal_rate",
            legend=False,
            label="%s" % (models[i]),
            ax=axs,
        )

    axs.set_title(f"Goal Rate")
    axs.set_xlabel("Shot probability model percentile")
    axs.set_ylabel("Goals / (Shots + Goals)")
    axs.set_xlim(left=101, right=-1)
    axs.set_ylim(bottom=0, top=1)
    vals = axs.get_yticks()
    axs.set_yticks(vals)
    axs.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    axs.legend()

    plt.show()
    fig.savefig(path_output_image)


def Visualize_Cumulative_Proportion(models, probs, y_true, path_output_image):
    bins = np.linspace(0, 95, num=20).astype(int)
    fig, axs = plt.subplots(1, 1)

    list_plot_points = list()
    for i in range(len(models)):
        perentiles = (np.argsort(np.argsort(probs[i])) + 1) * 100.0 / (len(probs[i]))
        perentile_bins = (np.digitize(perentiles, bins) - 1) * 5
        plot_points = pd.DataFrame(columns=["bin", "goal_rate", "goal_cum"])
        goal_cum = 0

    for j, bin in enumerate(np.flip(bins)):
        n_goal = np.sum(y_true[perentile_bins == bin])
        n_shot = np.sum(perentile_bins == bin)
        goal_rate = n_goal / n_shot
        goal_cum += n_goal / np.sum(y_true)
        plot_points.loc[j] = [bin, goal_rate, goal_cum]
    list_plot_points.append(plot_points)

    plot_points.loc[j + 1] = [100, 0, 0]
    sns.lineplot(
        data=plot_points,
        x="bin",
        y="goal_cum",
        legend=False,
        label="%s" % (models[i]),
        ax=axs,
    )

    axs.set_title(f"Cumulative % of Goal")
    axs.set_xlabel("Shot probability model percentile")
    axs.set_ylabel("Proportion")
    axs.set_xlim(left=105, right=-5)
    axs.set_ylim(bottom=0, top=1)
    vals = axs.get_yticks()
    axs.set_yticks(vals)
    axs.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    axs.legend()

    plt.show()
    fig.savefig(path_output_image)


def Visualize_Calibration(models, probs, y_true, path_output_image):
    fig = plt.figure()
    plt.title(f"Calibration Curve")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i in range(len(models)):
        prob_true, prob_pred = calibration_curve(y_true, probs[i], n_bins=20)
        plt.plot(prob_pred, prob_true, "s-", label="%s" % (models[i]))

    plt.xlabel("P(Pred)")
    plt.ylabel("P(Goal|Pred)")
    plt.legend()

    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.show()

    fig.savefig(path_output_image)

