from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
from scipy.stats import gmean
import os

EXCLUDED_PAPERS = ["2411.04715"]

class Evaluator:
    def __init__(self, file_dir="./output"):
        self.file_dir = file_dir
        self.results = []

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        gmean_f1 = gmean(f1)
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else None

        return {
            "size": len(y_pred),
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score_0": f1[0],
            "f1_score_1": f1[1],
            "macro_f1_score": macro_f1,
            "weighted_f1_score": weighted_f1,
            "gmean_f1_score": gmean_f1,
            "auc": auc
        }
    
    @staticmethod
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

    @staticmethod
    def dcg_at_k(relevance_scores, k):
        relevance_scores = np.array(relevance_scores)[:k]
        return np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))
    

    def ndcg_at_10(self, relevant_items, retrieved_items):
        ndcg_scores = []

        for relevant, retrieved in zip(relevant_items, retrieved_items):
            relevant_set = set(relevant)
            relevance_scores = [1 if item in relevant_set else 0 for item in retrieved[:10]]
            dcg = self.dcg_at_k(relevance_scores, 10)
            ideal_relevance_scores = sorted(relevance_scores, reverse=True)
            idcg = self.dcg_at_k(ideal_relevance_scores, 10)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores), ndcg_scores


    def evaluate_file(self, file_path):
        try:
            df_data = pd.read_json(file_path, dtype={"id": str})
            df_data = df_data.loc[~df_data.id.isin(EXCLUDED_PAPERS)].reset_index(drop=True)
            y_true = df_data["y_true"]
            y_pred = df_data["y_pred"]
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred)
            return metrics
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def evaluate_all_classifications(self, verbose=False):
        files = os.listdir(self.file_dir)
        files = [f for f in files if f.endswith(".json") and (f.startswith("exp_1") or f.startswith("exp_2"))]
        for filename in files:
            if verbose:
                print(f"Evaluating: {filename}")
            file_path = os.path.join(self.file_dir, filename)
            metrics = self.evaluate_file(file_path)
            if metrics:
                # Append metrics to the results DataFrame
                self.results.append({
                    "file_name": filename,
                    "size": metrics["size"],
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score_0": metrics["f1_score_0"],
                    "f1_score_1": metrics["f1_score_1"],
                    "macro_f1_score": metrics["macro_f1_score"],
                    "weighted_f1_score": metrics["weighted_f1_score"],
                    "gmean_f1_score": metrics["gmean_f1_score"],
                    "auc": metrics["auc"]
                })
        # Return the final results as a DataFrame
        return pd.DataFrame(self.results)

    def evaluate_all_recommendations(self, verbose=False):
        files = os.listdir(self.file_dir)
        files = [f for f in files if f.endswith(".json") and f.startswith("exp_3_")]

        for filename in files:
            if verbose:
                print(f"Evaluating: {filename}")
            file_path = os.path.join(self.file_dir, filename)

            try:
                df_data = pd.read_json(file_path, dtype={"id": str})
                df_data = df_data.loc[~df_data.id.isin(EXCLUDED_PAPERS)].reset_index(drop=True)
                ids = df_data["id"].tolist()
                start_ids = df_data["start_ids"].tolist()
                relevant_items = df_data["y_true"].apply(lambda x: x["title"]).tolist()
                retrieved_items = df_data["y_pred"].apply(lambda x: [y["title"] for y in x]).tolist()
                retrieved_items_scores = df_data["y_pred"].apply(lambda x: [y["score"] for y in x]).tolist()

                # Calculate metrics
                mrr, mrr_scores = self.calculate_mrr(relevant_items, retrieved_items)
                ndcg10, ndcg10_scores = self.ndcg_at_10(relevant_items, retrieved_items)

                # Append metrics to the results list
                self.results.append({
                    "file_name": filename,
                    "size": len(df_data),
                    "mrr": mrr,
                    "ndcg_at_10": ndcg10,
                    "mrr_scores": [
                        {"id": i, "start_id": s, "score": score}
                        for i, s, score in zip(ids, start_ids, mrr_scores)
                    ],
                    "ndcg_at_10_scores": [
                        {"id": i, "start_id": s, "score": score}
                        for i, s, score in zip(ids, start_ids, ndcg10_scores)
                    ]
                })

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

        # Return the final results as a DataFrame
        return pd.DataFrame(self.results)
    
    def save_json(self, df, file_name, file_path="./eval_output",  **kwargs):
        df.to_json(os.path.join(file_path, file_name), **kwargs)
        print(f"Results saved to {file_path}")

if __name__ == "__main__":
    save_config = {
        # "file_path": "./eval_output",
        "indent": 2,
        "orient": "records",
        "index": False,
    }
    evaluator = Evaluator(file_dir="./output")
    cls_res = evaluator.evaluate_all_classifications(verbose=True)
    evaluator.save_json(cls_res, "classification_summary.json", **save_config)
    # rec_res = evaluator.evaluate_all_recommendations(verbose=True)
    # evaluator.save_json(rec_res, "recommendation_summary.json", **save_config)
