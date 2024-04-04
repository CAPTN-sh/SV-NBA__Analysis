import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_statistics(clusters_stats, features, statistics=['mean'], figsize=(12, 8), title=None, xlabel='Cluster Label'):
    """
    Plot specified statistics for a list of features for each cluster.

    Parameters:
    - clusters_stats: dict, a dictionary containing descriptive statistics for each cluster.
    - features: list of str, the names of the features to plot.
    - statistics: list of str, the statistics to plot ('mean', 'median', 'min', 'max', etc.).
    - figsize: tuple, the size of the figure (width, height).
    - title: str, the title of the plot. If None, a default title will be generated.
    - xlabel: str, the label for the x-axis.
    """
    plt.figure(figsize=figsize)
    
    # Calculate the total number of bars for each cluster to determine offsets
    total_bars = len(features) * len(statistics)
    bar_width = 0.8 / total_bars  # Calculate bar width to fit all bars
    
    for feature in features:
        for statistic in statistics:
            # Extract the specified statistic for this feature from each cluster's statistics
            values = [clusters_stats[cluster_label].loc[feature, statistic] for cluster_label in clusters_stats]
            
            # Calculate position offset for each feature-statistic combination
            combo_index = features.index(feature) * len(statistics) + statistics.index(statistic)
            positions = np.arange(len(clusters_stats)) + (combo_index - total_bars / 2) * bar_width
            
            # Plot the bars for this feature-statistic combination
            plt.bar(positions, values, width=bar_width, label=f'{feature} ({statistic})')

    # Customize the plot
    plt.xticks(ticks=np.arange(len(clusters_stats)), labels=clusters_stats.keys())
    plt.xlabel(xlabel)
    
    if title is None:
        title = 'Statistics of Features for Each Cluster'
    plt.title(title)
    
    plt.legend()
    plt.grid(True, axis='y')  # Add grid lines for better readability
    plt.show()



def plot_losses(train_losses, eval_losses):
    """
    Plot the training and evaluation loss curves.

    Parameters:
    - train_losses: A list of training losses.
    - eval_losses: A list of evaluation losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', linestyle='-', color='blue')
    plt.plot(eval_losses, label='Eval Loss', marker='x', linestyle='--', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training and Evaluation')
    plt.legend()
    plt.grid(True)  # Optional, adds grid lines to the chart for easier reading
    # plt.ylim(0, 0.1)
    plt.show()


def plot_full_comparison(all_inputs, all_reconstructions, feature_names):
    
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(20, 6))

        # Plot that feature input for the entire dataset
        plt.plot(all_inputs[:, :, i].flatten(), label='Input', alpha=0.7)
        
        # Plot the reconstructed output of this feature for the entire dataset
        plt.plot(all_reconstructions[:, :, i].flatten(), label='Reconstructed', alpha=0.7, linestyle='--')

        plt.title(f'Input vs. Reconstructed {feature_name} (Entire Dataset)')
        plt.xlabel('Sample Index')
        plt.ylabel(feature_name)
        plt.legend()
        plt.show()

def plot_mean_std_feature(df, group_column, feature, title, xlabel, ylabel, save_path=None, std=True):
    """
    Plot a bar chart of the mean values for a selected feature with standard deviation as error bars.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    group_column (str): The column name used for grouping.
    feature (str): The name of the feature to be visualized.
    title (str): The title of the chart.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    save_path (str): The path where the plot will be saved. If None, the plot is not saved.
    """
    # Prepare the data
    mean_data = df.groupby(group_column)[feature].mean().reset_index()
    std_data = df.groupby(group_column)[feature].std().reset_index()

    # Create the plot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=group_column, y=feature, data=mean_data, palette='coolwarm', capsize=.1) # coolwarm, viridis
    
    # Add error bars for standard deviation
    if std:
        plt.errorbar(x=mean_data[group_column], y=mean_data[feature], yerr=std_data[feature], fmt='none', c='black', capsize=8, elinewidth=2)

    # Set the title and labels
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    
    # # Remove the top and right spines from plot
    # sns.despine()
    # Set all line widths for the axes to be bold
    ax = plt.gca()
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # # Display the numerical value above each bar
    # for index, value in enumerate(mean_data[feature]):
    #     plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

    # Show or save the plot based on 'save_path'
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the figure after saving to free up memory
    else:
        plt.tight_layout()
        plt.show()


def plot_cluster_counts(df, group_column, title, xlabel, ylabel, save_path=None):
    """
    Plot a bar chart showing the count of items in each cluster.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    group_column (str): The column name used for grouping to show the counts of.
    title (str): The title of the chart.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    save_path (str): The path where the plot will be saved. If None, the plot is not saved.
    """
    # Prepare the data: Calculate the count of items in each cluster
    count_data = df[group_column].value_counts().reset_index()
    count_data.columns = [group_column, 'count']

    # Sort the data by the group column to ensure the bars are in order
    count_data = count_data.sort_values(by=group_column)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=group_column, y='count', data=count_data, palette='coolwarm')  # You can choose any palette
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Remove the top and right spines from plot
    sns.despine()

    # Display the numerical value above each bar
    for index, row in count_data.iterrows():
        plt.text(row[group_column], row['count'], f'{row["count"]}', ha='center', va='bottom')

    # Show or save the plot based on 'save_path'
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the figure after saving to free up memory
    else:
        plt.tight_layout()
        plt.show()