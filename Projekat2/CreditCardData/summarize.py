import math
import pandas as pd
from pandas.plotting import scatter_matrix
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
row_names = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
total_cluster_num = 9


def read_data(file_path):
    # citanje fajla
    data = pd.read_csv(file_path)
    # pretprocesiranje podataka
    data = data.drop('CUST_ID', 1)
    data = data.fillna(0)
    data.apply(pd.to_numeric)
    return data


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
    print("Cluster number " + str(cluster_idx))
    f.write("Cluster number " + str(cluster_idx) + "\n")
    for col_name in column_names:
        print("\tCurrent column " + col_name)
        f.write("\tCurrent column " + col_name + "\n")
        for row_name in row_names:
            init_value = initial_stats_df[col_name][row_name]
            cluster_value = cluster_df[col_name][row_name]

            diff = init_value - cluster_value
            if diff < 0:
                print("\t\t" + row_name + " value higher in cluster by: " + str(abs(diff)))
                f.write("\t\t" + row_name + " value higher in cluster by: " + str(abs(diff)) + "\n")
            else:
                print("\t\t" + row_name + " value lower in cluster by: " + str(abs(diff)))
                f.write("\t\t" + row_name + " value lower in cluster by: " + str(abs(diff)) + "\n")
        print()
    print()
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

    print("Parallel box-plots created!")


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
        scatter_matrix(cluster, alpha=0.2, figsize=(6, 6), diagonal='kde')
        # Save the figure
        plt.savefig('matrices/' + 'matrix-cluster' + str(i) + '.png', bbox_inches='tight')
        i += 1


if __name__ == '__main__':

    initial_data = read_data(data_file_path)
    initial_data_stats = initial_statistics(initial_data)
    # mean, median, quartiles
    clusters_stats = summarize_all_clusters(total_cluster_num)
    compare_all_clusters(initial_data_stats, clusters_stats)
    # box-plots
    clusters_data = read_cluster_data(total_cluster_num)
    create_parallel_box_plots(initial_data, clusters_data)
    # 2D representation (every attribute with every attribute)
    create_2d_clusters(clusters_data)
