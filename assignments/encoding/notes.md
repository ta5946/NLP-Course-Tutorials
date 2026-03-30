# BPE Encoding Notes

## What We Did
Implemented Byte Pair Encoding from scratch in Python, covering:
- Building an initial character-level vocabulary from a corpus
- Counting adjacent symbol pair frequencies
- Iteratively merging the most frequent pairs
- Encoding new text using the learned merge rules

## What Happened
Training on a small corpus (`low`, `lower`, `lowest`, `newer`, `wider`) for 10 merges produced sensible subword tokens. Frequent words like `low` and `lower` were fully merged into single tokens, while rarer ones like `lowest` remained partially segmented due to the limited merge budget.

The `</w>` end-of-word marker occasionally appears as a standalone token when the final character of a word was never merged with it. This is expected BPE behavior, not a bug.
