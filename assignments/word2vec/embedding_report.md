# Word Embeddings Assignment Report

## Task 1: Word Similarity & Nearest Neighbours

### Results

The cosine similarity function was implemented from scratch using the formula
`cos(u, v) = (u · v) / (‖u‖ · ‖v‖)`, returning `0.0` for zero-norm vectors to avoid
division by zero. The nearest-neighbour search iterates over the full 400,000-word GloVe
vocabulary without using any built-in Gensim helpers.

Selected nearest neighbours (top-5):

| Query    | Top neighbours |
|----------|---------------|
| `king`   | prince, queen, son, brother, monarch |
| `doctor` | physician, nurse, dr., doctors, patient |
| `bank`   | banks, banking, credit, investment, financial |
| `python` | monty, php, perl, cleese, flipper |

### Polysemy: "bank"

The standalone `bank` embedding is dominated by the **financial** sense, with top
neighbours are *banks, banking, credit, investment, financial*. This reflects the much
higher frequency of the financial usage in the Wikipedia + Gigaword training corpus.

Averaging the vectors of `["river", "bank"]` clearly shifts the neighbourhood toward the
**geographical** sense: top neighbours become *banks, shore, rivers, along, flows*. This
demonstrates that static word embeddings conflate all senses of a word into a single
point, but simple vector arithmetic can partially steer the representation toward a
desired sense.

---

## Task 2: Word Analogy

### Classic Examples

The 3CosAdd formula `query = v(b) − v(a) + v(c)` was used throughout.

| Analogy | Top prediction |
|---------|---------------|
| man : king :: woman : ? | **queen** (0.783) ✓ |
| paris : france :: berlin : ? | **germany** (0.893) ✓ |
| walk : walked :: run : ? | went (0.734), *ran* was 2nd ✗ |
| slow : slower :: fast : ? | **faster** (0.803) ✓ |

### Benchmark Evaluation

**Accuracy: 83.3% (10 / 12 correct)**

| Category | Correct |
|----------|---------|
| Semantic – capitals | 3 / 3 |
| Semantic – royalty/gender | 3 / 3 |
| Syntactic – comparative/superlative | 2 / 3 |
| Syntactic – verb tense | 2 / 3 |

The two errors were:

- `big:bigger::small:?` → predicted *larger* instead of *smaller*. Both *larger* and
  *smaller* are valid comparatives, but the model generalises "bigger" to the concept of
  scale rather than the specific morphological pattern of *small*.
- `walk:walked::run:?` → predicted *went* instead of *ran*. The irregular past tense of
  *run* is harder to capture because *went* (past of *go*) shares a strong distributional
  context with *walked* and *ran*.

---

## Task 3: Visualising Embedding Space

Two projections were generated for 50 words across five semantic groups (*animals,
countries, professions, emotions, royalty*).

### PCA

PCA finds the directions of maximum global variance. The five groups show partial
separation, with *countries* and *royalty* most cleanly isolated and *emotions* and
*professions* overlapping more. Because PCA preserves global structure, distances between
clusters are meaningful.

### t-SNE

t-SNE optimises for local neighbourhood preservation and produces tighter, more visually
distinct clusters. All five groups are more clearly separated than in PCA, though
absolute inter-cluster distances are not interpretable.

### Reflection

**Q1: Do same-group words cluster together?**  
Yes in both projections, though more clearly in t-SNE. This confirms that GloVe captures
semantic similarity: words that appear in similar contexts end up close in vector space.

**Q2: Are antonyms close or far apart?**  
Antonym pairs such as *happy/sad* and *calm/anxious* are actually **close** in embedding
space. This is a well-known artefact: antonyms tend to appear in highly similar syntactic
and topical contexts ("I feel happy" / "I feel sad"), so distributional methods place them
nearby. Sentiment polarity is not directly encoded in standard word embeddings.

---

## Task 4: Sentence Representation via Averaging

### Results

| Method | Macro F1 | Spam Recall |
|--------|----------|-------------|
| Mean word embeddings | 0.8498 | 0.66 |
| TF-IDF (sklearn) | 0.8652 | 0.62 |

Both methods achieve around 95% overall accuracy on the SMS spam classification task,
with TF-IDF edging ahead slightly on macro F1 (0.865 vs 0.850).

### Discussion

**Q1: Does TF-IDF improve over mean embeddings?**  
Marginally yes on macro F1, but the difference is small. TF-IDF benefits from
high-precision spam keywords (e.g. *free*, *win*, *claim*) that receive high weights and
are rare in ham, driving its precision for spam to 0.98. However, its recall (0.62) is
lower than the embedding model's (0.66) because it cannot generalise to unseen or
paraphrased spam vocabulary. Mean embeddings generalise better semantically but lose
discriminative signal by averaging all tokens equally.

**Q2: When does averaging fail?**  
Averaging is insensitive to word order and negation. The sentences *"not happy"* and
*"happy"* produce nearly identical mean vectors because `not` has a generic embedding
that barely shifts the centroid. Analogously, *"the food was not bad"* and *"the food was
bad"* would be treated as similar, confusing a sentiment classifier.

**Q3: Representing "The bank was on the river bank"**  
A single averaged vector would conflate the two senses of *bank*. Better approaches
include: (a) contextualised embeddings (ELMo, BERT) which assign different vectors to
each occurrence depending on context; (b) sense disambiguation followed by sense-specific
embeddings; or (c) simply relying on TF-IDF-style term weights where neither sense needs
semantic generalisation.

---

## Task 5: Gender Bias in Embeddings

### Results

Bias score = mean cosine similarity to male-pole words − mean cosine similarity to
female-pole words. Positive = more male-associated.

| Profession | Bias score | Direction |
|------------|-----------|-----------|
| manager | +0.1554 | → male |
| president | +0.1325 | → male |
| engineer | +0.1285 | → male |
| secretary | +0.0865 | → male |
| scientist | +0.0811 | → male |
| programmer | +0.0750 | → male |
| lawyer | +0.0660 | → male |
| pilot | +0.0612 | → male |
| plumber | +0.0275 | → male |
| doctor | +0.0201 | → male |
| artist | +0.0191 | → male |
| teacher | +0.0113 | → male |
| chef | +0.0079 | ≈ neutral |
| librarian | −0.0299 | → female |
| nurse | −0.0945 | → female |
| receptionist | −0.1283 | → female |

### Discussion

**Q1: Strongest gender associations**  
*manager*, *president*, and *engineer* have the largest positive (male) scores; *nurse*
and *receptionist* have the largest negative (female) scores. Both ends reflect widely
documented occupational gender stereotypes present in text corpora.

**Q2: Surprising results**  
*secretary* scores positively (male-associated), despite being a profession
historically held predominantly by women. This may reflect that in formal news text the
word often refers to political titles such as *Secretary of State* or *Secretary General*,
roles that were male-dominated in the pre-2014 training corpus.

**Q3: Effect of the training corpus**  
The embeddings were trained on Wikipedia and Gigaword news text from 2014. Both sources
over-represent male subjects (politics, finance, sport) and under-represent female ones.
Professions described more frequently in the context of male pronouns will end up closer
to male-pole words. A corpus with more balanced gender representation, or one that
post-dates legislative and cultural shifts, would likely show different scores.

**Q4: Harmful vs. useful associations**  
Gendered associations are not always harmful. Linguistically, they can aid tasks such as
pronoun coreference resolution where grammatical gender is informative. They become
problematic when embedded in systems that make consequential decisions, such as CV
screening, loan approval, or medical diagnosis, where associating a profession with a
gender can perpetuate discrimination. The key distinction is between *descriptive*
associations (reflecting how language has historically been used) and *prescriptive* ones
(treating those associations as correct or desirable).
