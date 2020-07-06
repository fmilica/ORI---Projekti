from cluster import cluster_data
from summarize import summarize_data
from decision_tree import decision_tree_report

data_file_path = "data/credit_card_data.csv"

if __name__ == '__main__':
    # Klasterizacija podataka
    clustered_data_file_path = cluster_data(data_file_path)
    print("Data clusterization completed\n")
    # Generalna analiza rezultata
    # za ispis paralelnih matrica proslediti vrednost True
    # (kreiranje matrica dugo traje, pa je default vrednost False)
    summarize_data(data_file_path, False)
    print("Data summarization completed.\n")
    # DecisionTreeClassification analiza podataka
    decision_tree_report(clustered_data_file_path)
    print("Decision tree summarization completed\n")
    # The end
    print("Data successfully clustered and analysed!")
