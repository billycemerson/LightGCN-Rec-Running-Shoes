import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for training loss plotting
from data_preprocessing import TrainTestGenerator

logger = logging.getLogger(__name__)

# === Basic Evaluation Metrics ===

def compute_ranks(train, test, recommended):
    """Compute the rank of each test item in the recommended list, excluding training items"""
    train = set(train)
    recommended = [item for item in recommended if item not in train]
    ranks = []
    for item in test:
        try:
            rank = recommended.index(item) + 1  # 1-based indexing
        except ValueError:
            rank = None
        ranks.append(rank)
    return ranks

def compute_normalized_ranks(train, test, recommended):
    """Compute ranks and remove found items to simulate ideal top-N ranking"""
    train = set(train)
    recommended = [item for item in recommended if item not in train]
    ranks = []
    for item in test:
        try:
            rank = recommended.index(item) + 1
            recommended.pop(rank - 1)
        except (ValueError, IndexError):
            rank = None
        ranks.append(rank)
    return ranks

def hit_rate_at_k(ranks, k):
    ranks = pd.Series(ranks)
    hits = ranks[ranks.notna() & (ranks <= k)]
    return len(hits) / len(ranks) if len(ranks) > 0 else 0.0

def recall_at_k(ranks, k):
    return hit_rate_at_k(ranks, k)

def mean_reciprocal_rank(ranks):
    ranks = pd.Series(ranks).dropna()
    if len(ranks) == 0:
        return 0.0
    return (1 / ranks).mean()

def precision_at_k(ranks, k):
    ranks = pd.Series(ranks)
    hits = (ranks <= k).sum()
    return hits / (k * len(ranks)) if k > 0 else 0.0

def ndcg_at_k(ranks, k):
    ranks = pd.Series(ranks).dropna()
    if len(ranks) == 0:
        return 0.0
    ranks = ranks[(ranks > 0) & (ranks <= k)]
    if len(ranks) == 0:
        return 0.0
    dcg = (1 / np.log2(ranks + 1)).sum()
    n_rel = len(ranks)
    ideal_ranks = np.arange(1, min(n_rel, k) + 1)
    ideal_dcg = (1 / np.log2(ideal_ranks + 1)).sum()
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# === Stopwatch Timer ===

class Stopwatch:
    def __init__(self):
        self.start_times = {}
        self.times = {}

    def start(self, tag):
        self.start_times[tag] = time.time()

    def stop(self, tag):
        self.times[tag] = time.time() - self.start_times[tag]

    def get_df(self):
        return pd.DataFrame(self.times.items(), columns=["tag", "time"])

    def set_from_df(self, times_df):
        self.times = times_df.set_index("tag")["time"].to_dict()

# === Evaluator Class ===

class Evaluator:
    def __init__(self, model_init, train_test_generator: TrainTestGenerator):
        self.model_init = model_init
        self.train_test_generator = train_test_generator
        self.stopwatch = Stopwatch()
        self.results = pd.DataFrame()
        self.last_model = None  # Store the most recent trained model

    def evaluate(self):
        """
        Run forward-chaining training & evaluation per year.
        Track training loss and validation score for each epoch.
        Plot learning curves. Store all per-user recommendation results.
        """
        results = []
        self.loss_history = {}

        for test_year, train, test in self.train_test_generator.forward_chaining():
            logging.info(f"Test year: {test_year}")

            self.stopwatch.start(f"model_init_{test_year}")
            model = self.model_init()
            self.stopwatch.stop(f"model_init_{test_year}")

            self.stopwatch.start(f"model_fit_{test_year}")
            train_losses, val_scores = model.fit(train, test_year=test_year)
            self.stopwatch.stop(f"model_fit_{test_year}")

            self.loss_history[test_year] = {
                "loss": train_losses,
                "val": val_scores
            }

            self.last_model = model

            n_items = len(train["product_id"].unique())

            for user_id in test["user_id"].unique():
                user_train = list(train.loc[train["user_id"] == user_id, "product_id"])
                user_test = list(test[test["user_id"] == user_id].sort_values("timestamp")["product_id"])

                self.stopwatch.start(f"recommend_user_{test_year}_{user_id}")
                recommended = list(model.recommend(user_id, n_items))
                self.stopwatch.stop(f"recommend_user_{test_year}_{user_id}")

                ranks = compute_ranks(user_train, user_test, recommended)
                norm_ranks = compute_normalized_ranks(user_train, user_test, recommended)

                results_user = pd.DataFrame({
                    "user": user_id,
                    "item": user_test,
                    "ranks": ranks,
                    "norm_ranks": norm_ranks,
                    "test_year": test_year
                })
                results.append(results_user)

        self.results = pd.concat(results).reset_index(drop=True)

    def get_model(self):
        return self.last_model

    def _eval_by_year(self, metric_fn, col="ranks", Ks: list = None):
        if Ks is None:
            Ks = [5, 10, 20, 50]
        results = self.results
        results_df = []
        years = sorted(results["test_year"].unique())
        for year in years:
            results_year = results[results["test_year"] == year]
            row = [len(results_year)]
            for k in Ks:
                score = metric_fn(results_year[col], k)
                row.append(score)
            results_df.append(row)
        return pd.DataFrame(results_df, columns=["cases"] + Ks, index=years)

    def get_hit_rates(self, Ks: list = None):
        return self._eval_by_year(hit_rate_at_k, col="norm_ranks", Ks=Ks)

    def get_recalls(self, Ks: list = None):
        return self._eval_by_year(recall_at_k, col="ranks", Ks=Ks)

    def get_precisions(self, Ks: list = None):
        return self._eval_by_year(precision_at_k, col="ranks", Ks=Ks)

    def get_ndcgs(self, Ks: list = None):
        if Ks is None:
            Ks = [5, 10, 20, 50]
        results = []
        years = sorted(self.results["test_year"].unique())
        for year in years:
            year_data = self.results[self.results["test_year"] == year]
            row = [len(year_data)]
            for k in Ks:
                ndcg_scores = []
                for user, user_data in year_data.groupby("user"):
                    user_ranks = user_data["ranks"]
                    ndcg_scores.append(ndcg_at_k(user_ranks, k))
                row.append(np.mean(ndcg_scores) if ndcg_scores else 0.0)
            results.append(row)
        return pd.DataFrame(results, columns=["cases"] + Ks, index=years)

    def get_mrr(self):
        results = self.results
        results_df = []
        years = sorted(results["test_year"].unique())
        for year in years:
            results_year = results[results["test_year"] == year]
            ranks = results_year["norm_ranks"].dropna()
            row = [len(ranks), mean_reciprocal_rank(ranks)]
            results_df.append(row)
        return pd.DataFrame(results_df, columns=["cases", "mrr"], index=years)

    def get_raw_times(self):
        return self.stopwatch.get_df()

    def get_times(self):
        df = self.stopwatch.get_df()
        df["task"] = df["tag"].str.split("_").str[0:2].str.join("_")
        return df.groupby("task")["time"].describe()

    def get_fit_per_year_times(self):
        df = self.get_raw_times()
        df["task"] = df["tag"].str.split("_").str[0:2].str.join("_")
        df = df.set_index("task")
        return df.loc["model_fit"]

    def save_results(self, ranks_path=None, times_path=None):
        if ranks_path is not None:
            self.results.to_csv(ranks_path, index=False)
        if times_path is not None:
            self.get_raw_times().to_csv(times_path, index=False)

    def load_results(self, ranks_path=None, times_path=None):
        if ranks_path is not None:
            self.results = pd.read_csv(ranks_path)
        if times_path is not None:
            self.stopwatch.set_from_df(pd.read_csv(times_path))