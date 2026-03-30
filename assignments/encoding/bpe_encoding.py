"""
BPE Encoding Assignment (Student Version)
=========================================

Implement Byte Pair Encoding (BPE) training and tokenization.

Goal:
- Learn subword segmentation by repeatedly merging frequent symbol pairs.

Complete all TODOs.
"""

from collections import defaultdict


WordSymbols = tuple[str, ...]
Pair = tuple[str, str]


def build_initial_vocab(corpus: list[str]) -> dict[WordSymbols, int]:
    """
    Build initial BPE vocabulary from a corpus.

    Representation:
    - each word is split into characters
    - append end-of-word marker '</w>'
    - vocabulary stores word-symbol tuples with frequencies
    """
    # TODO: split each sentence into words and count frequencies
    word_freq: dict[str, int] = defaultdict(int)
    for sentence in corpus:
        for word in sentence.split():
            word_freq[word] += 1

    # TODO: convert each unique word into tuple(characters + '</w>')
    # TODO: return {word_symbols: frequency}
    return {tuple(list(word) + ["</w>"]): freq for word, freq in word_freq.items()}


def get_pair_counts(vocab: dict[WordSymbols, int]) -> dict[Pair, int]:
    """
    Count adjacent symbol pair frequencies over the vocabulary.
    """
    # TODO: for each tokenized word and its freq:
    # - iterate over adjacent symbol pairs
    # - accumulate pair counts weighted by word frequency
    pair_counts: dict[Pair, int] = defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts


def merge_pair(pair: Pair, vocab: dict[WordSymbols, int]) -> dict[WordSymbols, int]:
    """
    Merge one pair everywhere in the vocabulary.
    Example: ('l', 'o') -> 'lo'
    """
    # TODO: replace each exact adjacent occurrence of pair with merged token
    # Return a new vocabulary dictionary.
    merged_token = pair[0] + pair[1]
    new_vocab: dict[WordSymbols, int] = {}
    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


def train_bpe(
    corpus: list[str],
    num_merges: int,
) -> tuple[list[Pair], dict[WordSymbols, int]]:
    """
    Train BPE by performing `num_merges` merges.

    Returns:
    - merges in order
    - final vocabulary
    """
    vocab = build_initial_vocab(corpus)
    merges: list[Pair] = []

    # TODO: repeat up to num_merges:
    # - compute pair counts
    # - stop early if no pairs remain
    # - pick most frequent pair
    # - merge it in vocab
    # - append chosen pair to merges
    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)

    return merges, vocab


def encode_word(word: str, merges: list[Pair]) -> list[str]:
    """
    Encode a single word with learned merge rules.

    Start from characters + '</w>', then apply merges in learned order.
    """
    # TODO: initialize symbols from word characters + end marker
    symbols: list[str] = list(word) + ["</w>"]

    # TODO: for each merge rule in order:
    # - scan symbols left-to-right
    # - merge matching adjacent symbols
    # - keep non-matching symbols unchanged
    for pair in merges:
        merged_token = pair[0] + pair[1]
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    # optional: remove '</w>' from final output tokens if attached.
    return [token.replace("</w>", "") if token != "</w>" else token for token in symbols]


def encode_text(text: str, merges: list[Pair]) -> list[str]:
    """
    Encode a full text by concatenating encoded words.
    """
    # TODO: split text into words and encode each with encode_word
    tokens: list[str] = []
    for word in text.split():
        tokens.extend(encode_word(word, merges))
    return tokens


def main() -> None:
    corpus = [
        "low lower lowest",
        "newer wider",
        "low low lower",
    ]
    num_merges = 10

    merges, final_vocab = train_bpe(corpus, num_merges=num_merges)

    print("Learned merges:")
    for i, p in enumerate(merges, start=1):
        print(f"{i:02d}. {p[0]} + {p[1]}")

    print("\nFinal vocab entries:")
    for symbols, freq in sorted(final_vocab.items(), key=lambda x: (-x[1], x[0])):
        print(f"{symbols} -> {freq}")

    test_text = "lowest newer low"
    encoded = encode_text(test_text, merges)
    print("\nTest text:", test_text)
    print("Encoded tokens:", encoded)


if __name__ == "__main__":
    main()
