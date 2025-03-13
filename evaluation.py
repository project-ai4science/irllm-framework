from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os


def calculate_classification_resume(df_data):
    y_true = df_data["y_true"].to_numpy()
    y_pred = df_data["y_pred"].to_numpy()
    
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-Score:", f1_score(y_true, y_pred))
    print("Macro F1-Score:", f1_score(y_true, y_pred, average='macro'))

def evaluate_all_classifications():
    directory = os.fsencode("./output")
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith("classification"):
            print(filename)
            df_data = pd.read_csv(f"./output/{filename}")
            calculate_classification_resume(df_data)
            print()
            continue
        else:
            continue


def calculate_mrr(relevant_items, retrieved_items):
    reciprocal_ranks = []

    for relevant, retrieved in zip(relevant_items, retrieved_items):
        relevant_array = np.array(relevant)
        retrieved_array = np.array(retrieved)

        ranks = np.where(np.isin(retrieved_array, relevant_array))[0]
        if ranks.size > 0:
            first_relevant_rank = ranks[0] + 1
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks), reciprocal_ranks


def dcg_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores)[:k]
    return np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))

def ndcg_at_5(relevant_items, retrieved_items):
    ndcg_scores = []
    
    for relevant, retrieved in zip(relevant_items, retrieved_items):
        relevant_set = set(relevant)
        
        relevance_scores = [1 if item in relevant_set else 0 for item in retrieved[:5]]
        
        dcg = dcg_at_k(relevance_scores, 5)
        
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        idcg = dcg_at_k(ideal_relevance_scores, 5)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores), ndcg_scores