from re import escape, search
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data.generate_destinations import CATALOG, PRODUCTS, REGIONS
from src.common import load_json
from src.schemas import Candidate


def contains(text, phrase):
    return bool(search(r"(?<!\w)" + escape(phrase.casefold()) + r"(?!\w)", text.casefold()))


class Retriever:
    def __init__(self, destinations=None):
        destinations = load_json(CATALOG) if destinations is None else destinations
        self.total = len(destinations)
        self.rows = [row for row in destinations if row["active_status"]]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([
            f"{row['description']} {row['department']} {row['subcategory']} {' '.join(row['routing_keywords'])}"
            for row in self.rows
        ])
        self.by_department = {
            name: np.array([index for index, row in enumerate(self.rows) if row["department"] == name])
            for name in {row["department"] for row in self.rows}
        }

    def retrieve(self, text, department, k, region=None):
        if not 5 <= k <= 50:
            raise ValueError("Top-K must be between 5 and 50.")
        start = perf_counter()
        indices = self.by_department.get(department, np.array([], dtype=int))
        query = self.vectorizer.transform([text])
        scores = (self.matrix[indices] @ query.T).toarray().ravel()
        # Explicit product and region mentions help disambiguate otherwise identical queues.
        products = {product for product in PRODUCTS if contains(text, product)}
        regions = {region} if region else {name for name in REGIONS if contains(text, name)}
        ranking = scores + np.array([
            .30 * (row["product"] in products) + .25 * (row["region"] in regions)
            for row in (self.rows[i] for i in indices)
        ])
        candidates = []
        for index in np.argsort(-ranking, kind="stable")[:k]:
            row = self.rows[indices[index]]
            candidates.append(Candidate(**{field: row[field] for field in (
                "destination_id", "department", "subcategory", "description", "region", "product")},
                similarity=float(scores[index]), ranking_score=float(ranking[index])))
        return tuple(candidates), len(indices), (perf_counter() - start) * 1000
