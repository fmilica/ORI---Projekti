from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import pandas as pd
import math
import csv
import seaborn as sb

k_max = 4

data_file_path = "data/credit_card_data.csv"
column_names = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

column_new_names = ['BALANCE', 'BALANCE_FREQUENCY', 'ONEOFF_PURCHASES',
                    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE',
                    'CLUSTER_NUM']

'''
class Point:
    def __init__(self, balance, balanceFrequency, purchases,
                 oneOffPurchases, installmentsPurchases,
                 cashAdvance, purchasesFrequency,
                 oneOffPurchasesFrequency, purchasesInstallmentsFrequency,
                 cashAdvanceFrequency, cashAdvanceTrx,
                 purchasesTrx, creditLimit, payments,
                 minimumPayments, prcFullPayment, tenure):
        self.balance = balance
        self.balanceFrequency = balanceFrequency
        self.purchases = purchases
        self.oneOffPurchases = oneOffPurchases
        self.installmentsPurchases = installmentsPurchases
        self.cashAdvance = cashAdvance
        self.purchasesFrequency = purchasesFrequency
        self.oneOffPurchasesFrequency = oneOffPurchasesFrequency
        self.purchasesInstallmentsFrequency = purchasesInstallmentsFrequency
        self.cashAdvanceFrequency = cashAdvanceFrequency
        self.cashAdvanceTrx = cashAdvanceTrx
        self.purchasesTrx = purchasesTrx
        self.creditLimit = creditLimit
        self.payments = payments
        self.minimumPayments = minimumPayments
        self.prcFullPayment = prcFullPayment
        self.tenure = tenure
'''


class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []  # podaci koji pripadaju ovom klasteru

    def recalculate_center(self):
        # centar klastera se racuna kao prosecna vrednost svih podataka u klasteru
        new_center = [0 for i in range(len(self.center))]
        for d in self.data:
            for i in range(len(d)):
                new_center[i] += d[i]

        n = len(self.data)
        if n != 0:
            self.center = [x/n for x in new_center]


class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        """
        :param n_clusters: broj grupa (klastera)
        :param max_iter: maksimalan broj iteracija algoritma
        :return: None
        """
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data, normalize=True):
        self.data = data  # lista N-dimenzionalnih podataka
        if normalize:
            self.data = self.normalize_data(self.data)

        # kada algoritam zavrsi, u self.clusters treba da bude "n_clusters" klastera (tipa Cluster)
        dimensions = len(self.data[0])

        # napravimo N random tacaka i stavimo ih kao centar klastera
        for i in range(self.n_clusters):
            # point = self.data[int(random.uniform(0,len(self.data)))]
            point = [random.random() for x in range(dimensions)]
            self.clusters.append(Cluster(point))

        iter_no = 0
        not_moves = False
        while iter_no <= self.max_iter and (not not_moves):
            # ispraznimo podatke klastera
            for cluster in self.clusters:
                cluster.data = []

            for d in self.data:
                # index klastera kom pripada tacka
                cluster_index = self.predict(d)
                # dodamo tacku u klaster kako bi izracunali centar
                self.clusters[cluster_index].data.append(d)

            # preracunavanje centra
            not_moves = True
            for cluster in self.clusters:
                old_center = copy.deepcopy(cluster.center)
                cluster.recalculate_center()

                not_moves = not_moves and (cluster.center == old_center)

            print("Iter no: " + str(iter_no))
            iter_no += 1

    def predict(self, datum):
        # podatak pripada onom klasteru cijem je centru najblizi (po euklidskoj udaljenosti)
        # kao rezultat vratiti indeks klastera kojem pripada
        min_distance = None
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = self.euclidean_distance(datum, self.clusters[index].center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance

        return cluster_index

    def euclidean_distance(self, x, y):
        # euklidsko rastojanje izmedju 2 tacke
        sq_sum = 0
        for xi, yi in zip(x, y):
            sq_sum += (yi - xi)**2

        return sq_sum ** 0.5

    def normalize_data(self, data):
        # mean-std normalizacija
        cols = len(data[0])

        for col in range(cols):
            column_data = []
            for row in data:
                column_data.append(row[col])

            mean = np.mean(column_data)
            std = np.std(column_data)

            for row in data:
                row[col] = (row[col] - mean) / std

        return data

    def sum_squared_error(self):
        # SSE (sum of squared error)
        # unutar svakog klastera sumirati kvadrate rastojanja izmedju podataka i centra klastera
        sse = 0
        for cluster in self.clusters:
            for d in cluster.data:
                sse += self.euclidean_distance(cluster.center, d)

        return sse**2


def read_data(file_path):
    # citanje .csv fajla
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[0], axis=1)
    df = df.fillna(0)
    return df


def preprocess_data(df):
    # odbacivanje kolona sa velikom korelacijom
    df = df.drop(['PURCHASES', 'PURCHASES_FREQUENCY'], axis=1)

    # priprema podataka
    list_of_rows = [list(row) for row in df.values]
    list_of_rows = list_of_rows[1:]

    float_data = []
    for row in list_of_rows:
        new_row = []
        for value in row:
            float_value = float(value)
            if math.isnan(float_value):
                float_value = 0
            new_row.append(float_value)
        float_data.append(new_row)

    return float_data


def optimal_k_value(upper_limit, input_data):
    plt.figure()
    sum_squared_errors = []
    for n_clusters in range(2, upper_limit):
        print("Number of clusters: " + str(n_clusters))
        kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
        kmeans.fit(input_data, True)
        sse = kmeans.sum_squared_error()
        sum_squared_errors.append(sse)
    plt.plot(sum_squared_errors)
    plt.xlabel('# of clusters')
    plt.ylabel('SSE')
    plt.show()


def write_clusters(clusters):
    # dodavanje kolone sa brojem klastera kome pripadaju
    i = 0
    for cluster in clusters:
        for item in cluster.data:
            item.append(i)
        i += 1

    # upis u fajl pojedinacnih klastera
    for cluster in clusters:
        file_name = "data/clusters/cluster" + str(clusters.index(cluster)) + ".csv"
        with open(file_name, mode='w', newline='\n') as cluster_file:
            cluster_writer = csv.writer(cluster_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            cluster_writer.writerow(column_new_names)
            for row in cluster.data:
                cluster_writer.writerow(row)

    # upis u fajl celokupnih podataka sa oznakom klastera
    all_clusters = []
    for cluster in clusters:
        all_clusters.extend(cluster.data)
    file_name = "data/clustered_data.csv"
    with open(file_name, mode='w', newline='\n') as cluster_file:
        cluster_writer = csv.writer(cluster_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cluster_writer.writerow(column_new_names)
        for row in all_clusters:
            cluster_writer.writerow(row)
    print("Cluster files written")


def create_correlation_matrix(input_data, plot):
    # Analiza pocetnih podataka
    print("Heatmap")
    # Compute the correlation matrix
    corr = input_data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 15))

    # Generate a custom diverging colormap
    cmap = sb.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sb.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=False, annot=True)

    if plot:
        plt.show()

    # Save the figure
    fig.savefig("correlation-matrix.png", bbox_inches='tight')

    # Print all correlations above 0.8
    columns = list(corr)
    for i in columns:
        for col_name in column_names:
            if 0.8 <= corr[i][col_name] < 1:
                if plot:
                    print(col_name + "\t" + str(corr[i][col_name]))


if __name__ == "__main__":

    # Citanje fajla
    data_frame = read_data(data_file_path)

    # Kreiranje matrice korelacije
    #create_correlation_matrix(data_frame, False)

    # Pretprocesiranje
    data = preprocess_data(data_frame)
    print("Data preprocessed")

    # Odredjivanje optimalne k-vrednosti
    #optimal_k_value(k_max + 1, data)

    optimal_k = 4
    print("Optimal number of clusters: " + str(optimal_k))
    kmeans = KMeans(n_clusters=optimal_k, max_iter=100)
    kmeans.fit(data, True)

    write_clusters(kmeans.clusters)

