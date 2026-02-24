"""
Alternative Text Processing Assignment (Student Version)
=======================================================

Complete the TODO sections to build a full NLTK vs Stanza comparison pipeline.
This file is intentionally scaffolded for students.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk import ne_chunk, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

import stanza


TEXTS = [
    "Wait... did Dr. J. Smith (U.C. Berkeley) really say 'NLP is easy' at 3:30 p.m., or was it sarcasm?!",
    "Email me at first.last+nlp@uni-example.edu ASAP - unless you've already sent it via https://tinyurl.com/nlp-demo.",
    "The startup's Q4 revenue was $1.2M-ish (not audited), yet users wrote: 'app crashes on iOS17/Android14 :('",
    "I re-read the note: \"Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.\" Still parsing it...",
]


@dataclass
class PipelineResult:
    sentences: list[str]
    tokens: list[str]
    pos_tags: list[tuple[str, str]]
    lemmas: list[str]
    entities: list[tuple[str, str]]
    elapsed_s: float


class NLTKPipeline:
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()

        # TODO: Sentence tokenization using NLTK.
        sentences = sent_tokenize(text)

        # TODO: Word tokenization using NLTK.
        tokens = word_tokenize(text)

        # TODO: POS tagging using NLTK.
        pos_tags_result = pos_tag(tokens)

        # TODO: Lemmatize each token using WordNetLemmatizer.
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]

        # TODO: Named Entity Recognition with ne_chunk over POS tags.
        # Store entities as tuples: (entity_text, entity_label).
        entities: list[tuple[str, str]] = []
        chunked = ne_chunk(pos_tags_result)
        for subtree in chunked:
            if hasattr(subtree, "label"):
                entity_text = " ".join(word for word, tag in subtree.leaves())
                entity_label = subtree.label()
                entities.append((entity_text, entity_label))

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


class StanzaPipeline:
    def __init__(self) -> None:
        # TODO 5.1: Initialize a Stanza pipeline for English with correct arguments
        self.nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,lemma,ner",
            verbose=False,
        )

    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()

        # TODO: Run Stanza pipeline on text.
        doc = self.nlp(text)

        # TODO: Extract sentence texts.
        sentences = [sent.text for sent in doc.sentences]

        # TODO: Extract tokens (word text).
        tokens = [word.text for sent in doc.sentences for word in sent.words]

        # TODO: Extract POS tags as (word, tag).
        pos_tags_result = [
            (word.text, word.upos)
            for sent in doc.sentences
            for word in sent.words
        ]

        # TODO: Extract lemmas.
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]

        # TODO: Extract named entities as (entity_text, entity_type).
        entities = [(ent.text, ent.type) for ent in doc.entities]

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


def compare_counts(nltk_res: PipelineResult, stanza_res: PipelineResult) -> dict[str, int]:
    # TODO: Return a dictionary with these keys:
    # sentences_nltk, sentences_stanza, tokens_nltk, tokens_stanza,
    # entities_nltk, entities_stanza
    return {
        "sentences_nltk": len(nltk_res.sentences),
        "sentences_stanza": len(stanza_res.sentences),
        "tokens_nltk": len(nltk_res.tokens),
        "tokens_stanza": len(stanza_res.tokens),
        "entities_nltk": len(nltk_res.entities),
        "entities_stanza": len(stanza_res.entities),
    }


def visualize_token_counts(rows: list[dict[str, str]], output_path: Path) -> None:
    # TODO: Build labels: S1, S2, ... based on number of rows.
    labels = [f"S{i+1}" for i in range(len(rows))]

    # TODO: Read token counts from rows and convert to int.
    nltk_counts = [int(row["tokens_nltk"]) for row in rows]
    stanza_counts = [int(row["tokens_stanza"]) for row in rows]

    # TODO: Create a grouped bar chart (NLTK vs Stanza) and save it.
    # Requirements:
    # - X-axis: sample labels
    # - Y-axis: number of tokens
    # - title: "Token Count Comparison: NLTK vs Stanza"
    # - legend enabled
    # - save to output_path with dpi=150
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], nltk_counts, width, label="NLTK")
    ax.bar([i + width / 2 for i in x], stanza_counts, width, label="Stanza")

    ax.set_xlabel("Sample")
    ax.set_ylabel("Number of Tokens")
    ax.set_title("Token Count Comparison: NLTK vs Stanza")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_report(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Text Processing Comparison Report",
        "",
        "Fill in the analysis prompts under each sample after running the script.",
        "",
    ]

    for i, row in enumerate(rows, start=1):
        # TODO: Add a markdown block for each sample containing:
        # - sample heading
        # - source text
        # - timings
        # - token/entity counts
        # - quick inspection previews
        # - analysis prompts
        # Tip: use lines.extend([...])
        lines.extend([
            f"## Sample {i}",
            "",
            f"**Text:** {row['text']}",
            "",
            "### Timings",
            f"- NLTK: {row['nltk_time']}s",
            f"- Stanza: {row['stanza_time']}s",
            "",
            "### Counts",
            f"- Tokens — NLTK: {row['tokens_nltk']}, Stanza: {row['tokens_stanza']}",
            f"- Entities — NLTK: {row['entities_nltk']}, Stanza: {row['entities_stanza']}",
            "",
            "### Token Previews (first 12)",
            f"- NLTK: {row['nltk_tokens_preview']}",
            f"- Stanza: {row['stanza_tokens_preview']}",
            "",
            "### POS Tag Previews (first 8)",
            f"- NLTK: {row['nltk_pos_preview']}",
            f"- Stanza: {row['stanza_pos_preview']}",
            "",
            "### Named Entities",
            f"- NLTK: {row['nltk_entities']}",
            f"- Stanza: {row['stanza_entities']}",
            "",
            "### Analysis Prompts",
            "1. Which library segmented sentences more accurately for this sample? Why?",
            "2. Are there differences in tokenization (e.g. handling of punctuation, URLs, special chars)?",
            "3. Which library identified more/better named entities? Any false positives or misses?",
            "4. How do the POS tags compare? Note any disagreements.",
            "5. Comment on the speed difference. When would you prefer one library over the other?",
            "",
            "---",
            "",
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_nltk_resources() -> None:
    required = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
    ]
    for item in required:
        nltk.download(item, quiet=True)


def run() -> None:
    ensure_nltk_resources()
    stanza.download("en", verbose=False)

    nltk_pipe = NLTKPipeline()
    stanza_pipe = StanzaPipeline()

    report_rows: list[dict[str, str]] = []

    for text in TEXTS:
        nltk_res = nltk_pipe.process(text)
        stanza_res = stanza_pipe.process(text)
        counts = compare_counts(nltk_res, stanza_res)

        # TODO: Append a dictionary to report_rows containing:
        # text, nltk_time, stanza_time,
        # tokens_nltk, tokens_stanza,
        # entities_nltk, entities_stanza,
        # nltk_tokens_preview, stanza_tokens_preview,
        # nltk_pos_preview, stanza_pos_preview,
        # nltk_entities, stanza_entities
        # Use previews: first 12 tokens and first 8 POS tags.
        report_rows.append({
            "text": text,
            "nltk_time": f"{nltk_res.elapsed_s:.4f}",
            "stanza_time": f"{stanza_res.elapsed_s:.4f}",
            "tokens_nltk": str(counts["tokens_nltk"]),
            "tokens_stanza": str(counts["tokens_stanza"]),
            "entities_nltk": str(counts["entities_nltk"]),
            "entities_stanza": str(counts["entities_stanza"]),
            "nltk_tokens_preview": str(nltk_res.tokens[:12]),
            "stanza_tokens_preview": str(stanza_res.tokens[:12]),
            "nltk_pos_preview": str(nltk_res.pos_tags[:8]),
            "stanza_pos_preview": str(stanza_res.pos_tags[:8]),
            "nltk_entities": str(nltk_res.entities),
            "stanza_entities": str(stanza_res.entities),
        })

    report_path = Path("assignments/text_processing_alternative_report.md")
    plot_path = Path("assignments/token_count_comparison.png")

    # TODO: Call write_report and visualize_token_counts using report_rows.
    write_report(report_rows, report_path)
    visualize_token_counts(report_rows, plot_path)

    print("Student assignment executed.")
    print("Complete all TODO sections to generate full output files.")


if __name__ == "__main__":
    run()
