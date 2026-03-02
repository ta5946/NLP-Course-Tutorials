"""
Text Representation Techniques Assignment (Student Version)
===========================================================

Implement core text-representation methods for spam classification.

Covers:
- text preprocessing (tokenisation, normalisation, stop-word removal)
- vocabulary construction
- Bag-of-Words (Count Vectorizer) from scratch
- TF-IDF from scratch (with data-leakage-safe IDF)
- hand-crafted feature engineering

Complete all TODO sections.
"""
import math
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path("../data/spam.csv")

STOP_WORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "i me my we our you your he she it his her its they them their "
    "this that these those and but or nor so yet for of in on at to "
    "by with from up out about into through during before after above "
    "below between each few more most other some such no than then "
    "too very just here there what which who whom when where why how "
    "all both each few more most other some such only own same so "
    "than too s t ll re ve d m".split()
)


# ---------------------------------------------------------------------------
# Step 1 - Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load the SMS spam CSV and return (texts, labels) where label 1 = spam."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    texts = df["text"].astype(str).tolist()
    labels = (df["label"].str.lower() == "spam").astype(int).tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# Step 2 - Preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Normalise and tokenise a single SMS message.

    1. Lowercase.
    2. Remove punctuation and digits (keep alphabetic only).
    3. Split on whitespace.
    4. Optionally filter stop words.
    5. Filter single-character tokens.
    """
    # TODO: lowercase text
    text = text.lower()

    # TODO: remove punctuation and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # TODO: split into tokens
    tokens: list[str] = text.split()

    # TODO: if remove_stopwords is True, remove tokens from STOP_WORDS
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # TODO: remove single-character tokens
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


# ---------------------------------------------------------------------------
# Step 3 - Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(corpus: list[str], max_vocab: int = 3000) -> dict[str, int]:
    """
    Build a token -> index mapping from the training corpus.

    1. Preprocess every document.
    2. Count token frequencies.
    3. Keep the top max_vocab tokens.
    4. Return {token: index} ordered by descending frequency.
    """
    # TODO: create a vocabulary
    counter: Counter = Counter()
    for doc in corpus:
        tokens = preprocess(doc)
        counter.update(tokens)

    most_common = counter.most_common(max_vocab)
    vocab = {token: idx for idx, (token, _) in enumerate(most_common)}

    return vocab


# ---------------------------------------------------------------------------
# Step 4 - Bag-of-Words (Count Vectorizer)
# ---------------------------------------------------------------------------

def count_vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    """
    Convert texts to a count (Bag-of-Words) matrix.

    Returns numpy array of shape (n_docs, vocab_size), dtype int32.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)

    # TODO: for each document:
    # - preprocess document
    # - increment matrix[i, vocab[token]] for each token found in vocab
    for i, doc in enumerate(texts):
        tokens = preprocess(doc)
        for token in tokens:
            if token in vocab:
                matrix[i, vocab[token]] += 1

    return matrix


# ---------------------------------------------------------------------------
# Step 5 - TF-IDF
# ---------------------------------------------------------------------------

def compute_idf(corpus_tokens: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    """
    Compute the IDF vector for all vocabulary tokens.

    Formula (smoothed):
        IDF(t) = log((1 + N) / (1 + df(t))) + 1
    """
    n_docs = len(corpus_tokens)
    vocab_size = len(vocab)
    df = np.zeros(vocab_size, dtype=np.float64)

    # TODO: compute document frequency per token
    # Hint: use set(tokens) so each token counts once per document
    for tokens in corpus_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in vocab:
                df[vocab[token]] += 1

    # TODO: compute and return smoothed IDF vector
    idf = np.log((1 + n_docs) / (1 + df)) + 1
    return idf


def tfidf_vectorize(
    texts: list[str],
    vocab: dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """
    Convert texts to a TF-IDF matrix.

        TF(t, d)     = count(t, d) / len(d)
        TF-IDF(t, d) = TF(t, d) * IDF(t)

    IDF is passed in (pre-computed on training data) to prevent leakage.

    Returns numpy array of shape (n_docs, vocab_size), dtype float64.
    """
    n_docs = len(texts)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

    for i, doc in enumerate(texts):
        tokens = preprocess(doc)
        if not tokens:
            continue
        token_counts = Counter(tokens)
        doc_len = len(tokens)
        for token, count in token_counts.items():
            if token in vocab:
                tf = count / doc_len
                matrix[i, vocab[token]] = tf * idf[vocab[token]]

    return matrix


# ---------------------------------------------------------------------------
# Step 6 - Custom hand-crafted features
# ---------------------------------------------------------------------------

_SPAM_WORDS = frozenset(
    "free win winner won cash prize claim urgent call txt text reply "
    "guaranteed offer discount limited mobile ringtone download bonus "
    "selected congratulations awarded voucher reward collect".split()
)


def extract_custom_features(text: str) -> list[float]:
    """
    Hand-crafted feature vector for one SMS message (9 features).

    Suggested features:
    1. Message length (characters)
    2. Number of tokens
    3. Number of digits
    4. Number of uppercase characters
    5. Uppercase character ratio
    6. Number of punctuation marks (!, ?, .)
    7. Number of currency symbols ($, GBP, EUR)
    8. Count of spam-indicator words (lowercased)
    9. Type-token ratio (unique tokens / total tokens)
    """
    # TODO: compute all 9 features and return as list[float]
    # Tip: protect against division by zero for empty text/tokens.
    tokens_raw = text.split()
    num_tokens = len(tokens_raw)
    num_chars = len(text)

    num_digits = sum(c.isdigit() for c in text)
    num_upper = sum(c.isupper() for c in text)
    upper_ratio = num_upper / num_chars if num_chars > 0 else 0.0
    num_punct = sum(c in "!?." for c in text)
    num_currency = sum(c in "$\xa3\u20ac" for c in text) + text.lower().count("gbp") + text.lower().count("eur")

    text_lower = text.lower()
    spam_word_count = sum(1 for w in text_lower.split() if w in _SPAM_WORDS)

    unique_tokens = set(t.lower() for t in tokens_raw)
    ttr = len(unique_tokens) / num_tokens if num_tokens > 0 else 0.0

    return [
        float(num_chars),
        float(num_tokens),
        float(num_digits),
        float(num_upper),
        upper_ratio,
        float(num_punct),
        float(num_currency),
        float(spam_word_count),
        ttr,
    ]


# ---------------------------------------------------------------------------
# Step 7 - Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[int],
    y_test: list[int],
    label: str,
) -> None:
    """Train logistic regression and print a classification report."""
    # TODO: train LogisticRegression(max_iter=1000, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # TODO: predict on X_test
    y_pred = clf.predict(X_test)

    # TODO: compute macro F1
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # TODO: print section header and classification_report
    print(f"\n{'='*60}")
    print(f"  {label}  |  Macro F1: {macro_f1:.4f}")
    print('='*60)
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading dataset...")
    texts, labels = load_dataset(DATA_PATH)

    print(f"  Total samples : {len(texts)}")
    print(f"  Spam          : {sum(labels)}  ({sum(labels)/len(labels):.1%})")
    print(f"  Ham           : {len(labels)-sum(labels)}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )

    # TODO: build vocabulary from X_train_raw only (avoid leakage)
    vocab: dict[str, int] = build_vocabulary(X_train_raw, max_vocab=3000)

    # --- Bag-of-Words ---
    # TODO: vectorize train/test with count_vectorize and evaluate
    X_train_bow = count_vectorize(X_train_raw, vocab)
    X_test_bow  = count_vectorize(X_test_raw,  vocab)
    evaluate(X_train_bow, X_test_bow, y_train, y_test, "Bag-of-Words (Count Vectorizer)")

    # --- TF-IDF ---
    # TODO: preprocess X_train_raw -> tokens
    train_tokens = [preprocess(doc) for doc in X_train_raw]

    # TODO: compute idf on training tokens only
    idf = compute_idf(train_tokens, vocab)

    # TODO: vectorize train/test with tfidf_vectorize and evaluate
    X_train_tfidf = tfidf_vectorize(X_train_raw, vocab, idf)
    X_test_tfidf  = tfidf_vectorize(X_test_raw,  vocab, idf)
    evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF (from scratch)")

    # --- Custom features ---
    # TODO: build numpy feature matrices from extract_custom_features
    X_train_custom = np.array([extract_custom_features(t) for t in X_train_raw])
    X_test_custom  = np.array([extract_custom_features(t) for t in X_test_raw])

    # TODO: evaluate custom features
    evaluate(X_train_custom, X_test_custom, y_train, y_test, "Hand-crafted Features (9 features)")

    # --- Custom approach (optional) ---
    # TODO: add your own representation idea and evaluate it
    # (e.g., use scikit-learn tf-idf implementation and try to find the best hyperparameter values)
    # (run script using machine learning models)
    for ngram_range, min_df, sublinear_tf in [
        ((1, 1), 2, False),
        ((1, 2), 2, True),
        ((1, 3), 2, True),
    ]:
        vec = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            stop_words=list(STOP_WORDS),
            max_features=5000,
        )
        X_tr = vec.fit_transform(X_train_raw)
        X_te = vec.transform(X_test_raw)
        evaluate(
            X_tr, X_te, y_train, y_test,
            f"sklearn TF-IDF ngram={ngram_range} sublinear={sublinear_tf}"
        )


if __name__ == "__main__":
    main()
