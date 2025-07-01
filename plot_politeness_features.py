import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_politeness_features(features_df, word_counts=None):
    """
    Generate visualizations for politeness features.
    
    Parameters:
        features_df (pd.DataFrame): DataFrame where each row is a text, and columns are feature counts.
        word_counts (list or pd.Series, optional): Total word counts for each text, for normalization. 
                                                   Must be same length as number of rows in features_df.
    """
    if features_df.shape[0] == 1:
        print("Warning: Only one text found. Visualizations require multiple texts to compute variance.")
        return

    sns.set(style="whitegrid")
    color = "#4682B4"  # Steel Blue

    # Plot 1: Mean feature counts with standard error bars
    means = features_df.mean()
    ses = features_df.sem()

    # Sort by mean descending
    sorted_indices = means.sort_values(ascending=False).index
    means = means[sorted_indices]
    ses = ses[sorted_indices]

    plt.figure(figsize=(12, 6), dpi=150)
    x = np.arange(len(means))
    plt.bar(x, means.values, yerr=ses.values, capsize=5, color=color)

    plt.xticks(x, means.index, rotation=45, ha="right", fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel("Avg. count per text", fontsize=15)
    plt.title("Average Politeness Feature Counts", fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.grid(axis='x', visible=False)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().spines[['left', 'bottom']].set_linewidth(1.2)

    plt.tight_layout()
    plt.show()

    # Plot 2: Boxplot of raw counts per feature (sorted by overall mean)
    melted_df = features_df.melt(var_name="Feature", value_name="Feature Count per Text")
    melted_df['Feature'] = pd.Categorical(melted_df['Feature'], categories=means.index, ordered=True)

    plt.figure(figsize=(12, 6), dpi=150)
    sns.boxplot(x="Feature", y="Feature Count per Text", data=melted_df, color=color)

    plt.xticks(rotation=45, ha="right", fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel("Feature Count per Text", fontsize=15)
    plt.title("Distribution of Politeness Feature Counts", fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.grid(axis='x', visible=False)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().spines[['left', 'bottom']].set_linewidth(1.2)

    plt.tight_layout()
    plt.show()

    # Plot 3: Mean feature counts per 100 words
    if word_counts is not None:
        if len(word_counts) != len(features_df):
            raise ValueError("Length of word_counts must match number of rows in features_df")
            
        normed_df = features_df.divide(word_counts, axis=0) * 100
        normed_means = normed_df.mean()
        normed_ses = normed_df.sem()

        # Sort by normalized mean descending
        sorted_indices = normed_means.sort_values(ascending=False).index
        normed_means = normed_means[sorted_indices]
        normed_ses = normed_ses[sorted_indices]

        plt.figure(figsize=(12, 6), dpi=150)
        x = np.arange(len(normed_means))
        plt.bar(x, normed_means.values, yerr=normed_ses.values, capsize=5, color=color)

        plt.xticks(x, normed_means.index, rotation=45, ha="right", fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylabel("Avg. count per 100 words", fontsize=15)
        plt.title("Politeness Feature Use Normalized by Word Count", fontsize=16)

        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.grid(axis='x', visible=False)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.gca().spines[['left', 'bottom']].set_linewidth(1.2)

        plt.tight_layout()
        plt.show()
    else:
        print("Note: Skipped normalization plot (Plot 3) — provide word_counts to enable it.")
