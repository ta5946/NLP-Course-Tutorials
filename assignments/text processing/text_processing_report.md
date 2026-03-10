# Text Processing Comparison Report

Fill in the analysis prompts under each sample after running the script.

## Sample 1

**Text:** Wait... did Dr. J. Smith (U.C. Berkeley) really say 'NLP is easy' at 3:30 p.m., or was it sarcasm?!

### Timings
- NLTK: 3.8849s
- Stanza: 0.4676s

### Counts
- Tokens — NLTK: 27, Stanza: 26
- Entities — NLTK: 2, Stanza: 3

### Token Previews (first 12)
- NLTK: ['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C', '.', 'Berkeley', ')', 'really']
- Stanza: ['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C.', 'Berkeley', ')', 'really', 'say']

### POS Tag Previews (first 8)
- NLTK: [('Wait', 'NN'), ('...', ':'), ('did', 'VBD'), ('Dr.', 'NNP'), ('J.', 'NNP'), ('Smith', 'NNP'), ('(', '('), ('U.C', 'NNP')]
- Stanza: [('Wait', 'VERB'), ('...', 'PUNCT'), ('did', 'AUX'), ('Dr.', 'PROPN'), ('J.', 'PROPN'), ('Smith', 'PROPN'), ('(', 'PUNCT'), ('U.C.', 'PROPN')]

### Named Entities
- NLTK: [('Wait', 'GPE'), ('Berkeley', 'PERSON')]
- Stanza: [('J. Smith', 'PERSON'), ('U.C. Berkeley', 'ORG'), ('3:30 p.m.', 'TIME')]

### Analysis

1. **Which library segmented sentences more accurately for this sample? Why?**
   Both correctly identified this as a single sentence. No difference observed.

2. **Are there differences in tokenization (e.g. handling of punctuation, URLs, special chars)?**
   Yes, one difference: NLTK split `U.C.` into two tokens — `U.C` and `.` — while Stanza kept it as a single token `U.C.`. This accounts for the token count difference (27 vs 26). All other tokens in the preview are identical between the two libraries.

3. **Which library identified more/better named entities? Any false positives or misses?**
   Stanza performed significantly better. NLTK produced two false positives — `Wait` labeled as GPE and `Berkeley` labeled as PERSON — missing that `Berkeley` belongs with `U.C.` as part of the organization name. Stanza correctly identified `J. Smith` as PERSON, `U.C. Berkeley` as ORG, and `3:30 p.m.` as TIME, none of which NLTK caught.

4. **How do the POS tags compare? Note any disagreements.**
   The tagsets are different (Penn Treebank vs Universal Dependencies) so direct comparison requires mapping. Within the visible preview: NLTK tagged `Wait` as `NN` (noun) while Stanza tagged it as `VERB`. NLTK tagged `did` as `VBD` while Stanza tagged it as `AUX`. Both tagged `Dr.`, `J.`, and `Smith` as proper nouns (different notation: `NNP` vs `PROPN`).

5. **Comment on the speed difference. When would you prefer one library over the other?**
   NLTK took 3.88s vs Stanza's 0.47s, but this is due to NLTK's lazy model loading on the first call — samples 2–4 show NLTK consistently at ~0.38s, on par with Stanza. The timing difference here is a one-time warmup cost, not a per-text difference.

---

## Sample 2

**Text:** Email me at first.last+nlp@uni-example.edu ASAP - unless you've already sent it via https://tinyurl.com/nlp-demo.

### Timings
- NLTK: 0.3871s
- Stanza: 0.5036s

### Counts
- Tokens — NLTK: 19, Stanza: 14
- Entities — NLTK: 1, Stanza: 0

### Token Previews (first 12)
- NLTK: ['Email', 'me', 'at', 'first.last+nlp', '@', 'uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already']
- Stanza: ['Email', 'me', 'at', 'first.last+nlp@uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already', 'sent', 'it']

### POS Tag Previews (first 8)
- NLTK: [('Email', 'VB'), ('me', 'PRP'), ('at', 'IN'), ('first.last+nlp', 'JJ'), ('@', 'JJ'), ('uni-example.edu', 'JJ'), ('ASAP', 'NNP'), ('-', ':')]
- Stanza: [('Email', 'VERB'), ('me', 'PRON'), ('at', 'ADP'), ('first.last+nlp@uni-example.edu', 'PROPN'), ('ASAP', 'ADV'), ('-', 'PUNCT'), ('unless', 'SCONJ'), ('you', 'PRON')]

### Named Entities
- NLTK: [('ASAP', 'ORGANIZATION')]
- Stanza: []

### Analysis

1. **Which library segmented sentences more accurately for this sample?**
   Both correctly identified this as a single sentence. No difference observed.

2. **Are there differences in tokenization (e.g. handling of punctuation, URLs, special chars)?**
   Yes, this is the largest tokenization difference in the dataset (19 vs 14 tokens). NLTK split the email address into three tokens — `first.last+nlp`, `@`, `uni-example.edu` — while Stanza kept it as one token `first.last+nlp@uni-example.edu`. The URL at the end of the sentence also accounts for additional NLTK tokens not visible in the 12-token preview. Stanza treated both as single units.

3. **Which library identified more/better named entities? Any false positives or misses?**
   NLTK labeled `ASAP` as ORGANIZATION — a false positive. Stanza found no entities. Since there are no true named entities in this sentence, Stanza's empty result is the correct outcome.

4. **How do the POS tags compare? Note any disagreements.**
   Due to NLTK's email fragmentation, its POS tags for those tokens are unreliable — it fell back to `JJ` (adjective) for `first.last+nlp`, `@`, and `uni-example.edu`, which are clearly wrong. Stanza tagged the whole email as `PROPN`. For `ASAP`, NLTK assigned `NNP` (proper noun) while Stanza assigned `ADV` (adverb). Both tagged `Email` as a verb and `me` as a pronoun.

5. **Comment on the speed difference. When would you prefer one library over the other?**
   Both are roughly equal after warmup (0.39s vs 0.50s). No meaningful difference at this text length.

---

## Sample 3

**Text:** The startup's Q4 revenue was $1.2M-ish (not audited), yet users wrote: 'app crashes on iOS17/Android14 :('

### Timings
- NLTK: 0.3812s
- Stanza: 0.5117s

### Counts
- Tokens — NLTK: 24, Stanza: 27
- Entities — NLTK: 0, Stanza: 1

### Token Previews (first 12)
- NLTK: ['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2M-ish', '(', 'not', 'audited', ')']
- Stanza: ['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2', 'M', '-', 'ish', '(']

### POS Tag Previews (first 8)
- NLTK: [('The', 'DT'), ('startup', 'NN'), ("'s", 'POS'), ('Q4', 'NNP'), ('revenue', 'NN'), ('was', 'VBD'), ('$', '$'), ('1.2M-ish', 'JJ')]
- Stanza: [('The', 'DET'), ('startup', 'NOUN'), ("'s", 'PART'), ('Q4', 'PROPN'), ('revenue', 'NOUN'), ('was', 'AUX'), ('$', 'SYM'), ('1.2', 'NUM')]

### Named Entities
- NLTK: []
- Stanza: [('$1.2M', 'MONEY')]

### Analysis

1. **Which library segmented sentences more accurately for this sample?**
   Both correctly identified this as a single sentence. No difference observed.

2. **Are there differences in tokenization (e.g. handling of punctuation, URLs, special chars)?**
   Yes. NLTK kept `1.2M-ish` as a single token, while Stanza split it into four tokens: `1.2`, `M`, `-`, `ish`. This is the main driver of the token count difference (24 vs 27). Neither approach is objectively correct — `1.2M-ish` is an informal unit and splitting it is a reasonable choice, but so is keeping it whole.

3. **Which library identified more/better named entities? Any false positives or misses?**
   Stanza found `$1.2M` as MONEY; NLTK found nothing. Stanza's NER reconstructed the entity across the `$`, `1.2`, and `M` tokens it had split apart. Neither library identified `Q4`, `iOS17`, or `Android14`.

4. **How do the POS tags compare? Note any disagreements.**
   Visible differences in the preview: `was` is tagged `VBD` by NLTK and `AUX` by Stanza. `$` is tagged `$` by NLTK and `SYM` by Stanza — different notation, same idea. `1.2M-ish` is tagged `JJ` (adjective) by NLTK, while Stanza tags `1.2` as `NUM` after splitting. Both agree on `The` (DT/DET), `startup` (NN/NOUN), `'s` (POS/PART), `Q4` (NNP/PROPN), and `revenue` (NN/NOUN).

5. **Comment on the speed difference. When would you prefer one library over the other?**
   Essentially identical after warmup (0.38s vs 0.51s). No meaningful difference at this text length.

---

## Sample 4

**Text:** I re-read the note: "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo." Still parsing it...

### Timings
- NLTK: 0.3962s
- Stanza: 0.3801s

### Counts
- Tokens — NLTK: 20, Stanza: 20
- Entities — NLTK: 3, Stanza: 3

### Token Previews (first 12)
- NLTK: ['I', 're-read', 'the', 'note', ':', '``', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo']
- Stanza: ['I', 're-read', 'the', 'note', ':', '"', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo']

### POS Tag Previews (first 8)
- NLTK: [('I', 'PRP'), ('re-read', 'VBP'), ('the', 'DT'), ('note', 'NN'), (':', ':'), ('``', '``'), ('Buffalo', 'NNP'), ('buffalo', 'NN')]
- Stanza: [('I', 'PRON'), ('re-read', 'VERB'), ('the', 'DET'), ('note', 'NOUN'), (':', 'PUNCT'), ('"', 'PUNCT'), ('Buffalo', 'PROPN'), ('buffalo', 'PROPN')]

### Named Entities
- NLTK: [('Buffalo', 'PERSON'), ('Buffalo', 'PERSON'), ('Buffalo', 'PERSON')]
- Stanza: [('Buffalo', 'GPE'), ('Buffalo', 'GPE'), ('Buffalo', 'GPE')]

### Analysis

1. **Which library segmented sentences more accurately for this sample?**
   Both correctly split this into two sentences (`I re-read the note: "Buffalo buffalo..."` and `Still parsing it...`). No difference observed.

2. **Are there differences in tokenization (e.g. handling of punctuation, URLs, special chars)?**
   Token counts are identical (20). The only difference is the opening quotation mark: NLTK converted `"` to ` `` ` (Penn Treebank backtick convention), while Stanza kept the original `"` character. All other tokens are the same.

3. **Which library identified more/better named entities? Any false positives or misses?**
   Both found 3 entities, but with different labels: NLTK labeled them as PERSON, Stanza as GPE (geopolitical entity, i.e. a city). Since Buffalo is a city, GPE is the correct label. Both libraries missed the other five occurrences of `buffalo` in the sentence.

4. **How do the POS tags compare? Note any disagreements.**
   In the visible preview, NLTK tagged `buffalo` (lowercase) as `NN` (common noun) while Stanza tagged it as `PROPN` (proper noun). For the first `Buffalo` (capitalized), both tagged it as a proper noun (`NNP` vs `PROPN`). Everything else in the preview matches: `I` (PRP/PRON), `re-read` (VBP/VERB), `the` (DT/DET), `note` (NN/NOUN), `:` (:/ PUNCT).

5. **Comment on the speed difference. When would you prefer one library over the other?**
   Essentially identical (0.40s vs 0.38s) — the only sample where Stanza was marginally faster. At this text length the difference is noise.

---
