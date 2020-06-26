import math
import pandas as pd
import matplotlib as mpl
# used for creating plot .png file
mpl.use('agg')
import matplotlib.pyplot as plt

# static variables
data_file_path = "data/credit_card_data.csv"
column_names = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
row_names = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
total_cluster_num = 7


def initial_statistics(file_path):
    # citanje fajla
    credit_card_data = pd.read_csv(file_path)
    # pretprocesiranje podataka
    credit_card_data = credit_card_data.drop('CUST_ID', 1)
    credit_card_data = credit_card_data.fillna(0)
    credit_card_data.apply(pd.to_numeric)

    initial_stats = credit_card_data.describe()
    initial_stats.to_csv('data/summarize/initial_info.csv')
    return initial_stats


def summarize_all_clusters(cluster_number):
    all_cluster_stats = []
    for cluster_idx in range(0, cluster_number):
        cluster_df = pd.read_csv("data/cluster" + str(cluster_idx) + ".csv")
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
    print("Cluster number " + str(cluster_idx))
    for col_name in column_names:
        print("\tCurrent column " + col_name)
        for row_name in row_names:
            init_value = initial_stats_df[col_name][row_name]
            cluster_value = cluster_df[col_name][row_name]

            diff = init_value - cluster_value
            if diff < 0:
                print("\t\t" + row_name + " value higher in cluster by: " + str(abs(diff)))
            else:
                print("\t\t" + row_name + " value lower in cluster by: " + str(abs(diff)))
        print()
    print()


def read_cluster_data(cluster_number):
    all_clusters_stats = []
    for cluster_idx in range(0, cluster_number):
        cluster_df = pd.read_csv("data/cluster" + str(cluster_idx) + ".csv")
        all_clusters_stats.append(cluster_df)
    return all_clusters_stats


def preprocces_data(col_cluster_data):
    proccessed_col_data = []
    for value in col_cluster_data:
        proccessed_col_data.append(float(value))
    return proccessed_col_data


def create_parallel_box_plots(all_cluster_data):
    # paralelni box-plotovi za svaku kolonu
    for col_name in column_names:
        col_data_to_plot = []
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
    xticklabels = []
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


if __name__ == '__main__':

    initial_data_stats = initial_statistics(data_file_path)
    clusters_stats = summarize_all_clusters(total_cluster_num)
    compare_all_clusters(initial_data_stats, clusters_stats)
    # box-plots
    clusters_data = read_cluster_data(total_cluster_num)
    create_parallel_box_plots(clusters_data)
