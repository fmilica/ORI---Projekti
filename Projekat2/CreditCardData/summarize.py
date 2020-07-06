import math
import random
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
# used for creating plot .png file
mpl.use('agg')

# static variables
data_file_path = "data/credit_card_data.csv"
column_names = ['BALANCE', 'BALANCE_FREQUENCY', 'ONEOFF_PURCHASES',
                'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
all_column_names = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
row_names = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
total_cluster_num = 4

# determine lowest and highest values per column
cluster_values = {column: [] for column in column_names}
cluster_min_max = {cluster: [] for cluster in range(total_cluster_num)}


def read_data(file_path):
    # citanje fajla
    data = pd.read_csv(file_path)
    # pretprocesiranje podataka
    data = data.drop('CUST_ID', 1)
    data = data.fillna(0)
    data.apply(pd.to_numeric)
    return data


def normalize_data(data):
    normalized = StandardScaler().fit_transform(data)
    normalized_df = pd.DataFrame(normalized, columns=all_column_names)
    return normalized_df


def initial_statistics(credit_card_data):
    # bazna statistika
    initial_stats = credit_card_data.describe()
    initial_stats.to_csv('data/sum_initial.csv')
    return initial_stats


def summarize_all_clusters(cluster_number):
    all_cluster_stats = []
    for cluster_idx in range(0, cluster_number):
        cluster_df = pd.read_csv("data/clusters/cluster" + str(cluster_idx) + ".csv")
        cluster_stats = summarize_cluster(cluster_idx, cluster_df)
        all_cluster_stats.append(cluster_stats)
    return all_cluster_stats


def summarize_cluster(cluster_idx, cluster_df):
    cluster_stats = cluster_df.describe()
    cluster_stats.to_csv('data/summarize/sum_cluster' + str(cluster_idx) + '.csv')
    return cluster_stats


def compare_all_clusters(initial_stats_df, all_cluster_stats):
    idx = 0
    for cluster_stats in all_cluster_stats:
        compare_cluster(initial_stats_df, cluster_stats, idx)
        idx += 1


def compare_cluster(initial_stats_df, cluster_df, cluster_idx):
    f = open("data/sum-text/desc_cluster" + str(cluster_idx) + ".txt", "w")
    f.write("Cluster number " + str(cluster_idx) + "\n")
    for col_name in column_names:
        f.write("\tCurrent column " + col_name + "\n")
        for row_name in row_names:
            init_value = initial_stats_df[col_name][row_name]
            cluster_value = cluster_df[col_name][row_name]

            # append to lowest and highest dictionaries
            if row_name == "50%":
                cluster_values[col_name].append(cluster_value)

            diff = init_value - cluster_value
            if diff < 0:
                f.write("\t\t" + row_name + " value higher in cluster than in initial dataset by: "
                        + str(abs(diff)) + "\n")
            else:
                f.write("\t\t" + row_name + " value lower in cluster than in initial dataset by: "
                        + str(abs(diff)) + "\n")
    f.close()


def read_cluster_data(cluster_number):
    all_clusters_stats = []
    for cluster_idx in range(0, cluster_number):
        cluster_df = pd.read_csv("data/clusters/cluster" + str(cluster_idx) + ".csv")
        all_clusters_stats.append(cluster_df)
    return all_clusters_stats


def preprocces_data(col_cluster_data):
    proccessed_col_data = []
    for value in col_cluster_data:
        proccessed_col_data.append(float(value))
    return proccessed_col_data


def create_parallel_box_plots(start_data, all_cluster_data):
    # paralelni box-plotovi za svaku kolonu
    for col_name in column_names:
        col_data_to_plot = []
        # pretprocesiranje pocetnih podataka (lista lista)
        col_start_data = start_data[col_name]
        start_data_per_col = preprocces_data(col_start_data)
        col_data_to_plot.append(start_data_per_col)
        for cluster_data in all_cluster_data:
            col_cluster_data = cluster_data[col_name]
            # pretprocesiranje
            # treba nam lista lista
            cluster_data_per_col = preprocces_data(col_cluster_data)
            col_data_to_plot.append(cluster_data_per_col)
        plot_data(col_data_to_plot, total_cluster_num, col_name)


def plot_data(data_to_plot, number_of_clusters, column_name):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Custom x-axis labels
    xticklabels = ['Initial']
    for cluster_num in range(0, number_of_clusters):
        xticklabels.append('Cluster' + str(cluster_num))
    ax.set_xticklabels(xticklabels)

    # Custom y-axis label
    ax.set_ylabel(column_name)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    fig.savefig('figures/' + column_name.lower() + '.png', bbox_inches='tight')

    # Closes plot
    plt.clf()


def create_2d_clusters(all_cluster_data):
    # Create scatter matrices
    i = 0
    for cluster in all_cluster_data:
        cluster = cluster.iloc[:, :-1]
        scatter_matrix(cluster, alpha=0.2, figsize=(6, 6), diagonal='kde')
        # Save the figure
        plt.savefig('matrices/' + 'matrix-cluster' + str(i) + '.png')
        i += 1
        print("Created " + str(i) + ". matrix.")


def pca_2d_clusters(all_cluster_data):
    # Create scatter plots
    i = 0
    targets = []
    final_dfs = []
    for cluster in all_cluster_data:
        # Add cluster index to target
        targets.append("cluster" + str(i))

        # Separating out the features
        x = cluster.loc[:, column_names].values
        data_length = np.shape(x)[0]
        # Creating target array
        y = np.full((1, data_length), "cluster" + str(i))
        # Standardizing the features
        x = StandardScaler().fit_transform(x)

        # Reducing dimensions
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components,
                                   columns=['principal component 1', 'principal component 2'])
        # Add cluster number attribute
        cluster_number_col = pd.DataFrame({'CLUSTER': y[0]})
        final_df = pd.concat([principal_df, cluster_number_col], axis=1)
        final_dfs.append(final_df)
        # increment index
        i += 1

    # concatenate all data frames
    full_df = pd.concat(final_dfs)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    # generate colors
    cm = pylab.get_cmap('gist_rainbow')
    colors = []
    for i in range(total_cluster_num):
        color = cm(1. * i / total_cluster_num)
        colors.append(color)

    for target, c in zip(targets, colors):
        indices_to_keep = full_df['CLUSTER'] == target
        ax.scatter(full_df.loc[indices_to_keep, 'principal component 1'],
                   full_df.loc[indices_to_keep, 'principal component 2'],
                   c=c,
                   s=50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('plots/2d_clusters.png', bbox_inches='tight')


def describe_clusters():
    # Find highest and lowest values in all clusters per column
    for col_name, value_list in cluster_values.items():
        minimum = min(value_list)
        min_idxs = [i for i, x in enumerate(value_list) if x == minimum]
        if len(min_idxs) == 1:
            cluster_min_max[min_idxs[0]].append((col_name, minimum, 'min'))
        maximum = max(value_list)
        max_idxs = [i for i, x in enumerate(value_list) if x == maximum]
        if len(max_idxs) == 1:
            cluster_min_max[max_idxs[0]].append((col_name, maximum, 'max'))

    # Write characteristics of each cluster to file
    f = open("data/conclusions/conclusion.txt", "w")
    for cluster_idx, tuple_list in cluster_min_max.items():
        f.write("\nCluster" + str(cluster_idx) + ":\n")
        for tuple_value in tuple_list:
            col_name, value, type = tuple_value
            if type == 'min':
                f.write("\tMinimum overall value of " + col_name + ": " + str(value) + "\n")
            else:
                f.write("\tMaximum overall value of " + col_name + ": " + str(value) + "\n")
    f.close()


def summarize_data(file_path=data_file_path, matrices=False):
    # initial data statistics
    initial_data = read_data(file_path)
    initial_data_stats = initial_statistics(initial_data)
    normalized_initial_data_stats = initial_statistics(normalize_data(initial_data))
    # mean, median, quartiles
    clusters_stats = summarize_all_clusters(total_cluster_num)
    print("Basic statistical analysis create")

    compare_all_clusters(normalized_initial_data_stats, clusters_stats)
    # clusters with highest/lowest values
    describe_clusters()
    print("Cluster statistics compared")

    # box-plots
    clusters_data = read_cluster_data(total_cluster_num)
    create_parallel_box_plots(normalize_data(initial_data), clusters_data)
    print("Parallel box-plots created")

    # 2D representation (using PCA algorithm to reduce dimensionality)
    pca_2d_clusters(clusters_data)

    # 2D representation (every attribute with every attribute)
    if matrices:
        create_2d_clusters(clusters_data)
        print("All cluster matrices created")

    print("Summarization completed")
