"""Model 1 (technique area B): classical baseline.
tf-idf features (word unigrams + bigrams) -> logistic regression.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from router.data import load_splits

splits = load_splits()
x = [t for t, _ in splits["train"]]
y = [l for _, l in splits["train"]]

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
    LogisticRegression(C=20, max_iter=3000),
)
model.fit(x, y)
va = splits["val"]
acc = sum(p == l for p, (_, l) in zip(model.predict([t for t, _ in va]), va)) / len(va)
print(f"validation accuracy {acc:.4f}")
joblib.dump(model, "models/tfidf_logreg.joblib")
