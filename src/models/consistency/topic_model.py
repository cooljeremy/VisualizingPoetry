from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def topic_vector(texts: List[str], k: int = 10):
    vec = CountVectorizer(max_features=5000)
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=42)
    dist = lda.fit_transform(X)
    return dist.tolist()


