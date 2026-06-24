#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_STOPWORDS = frozenset(
    """
    a about above across after afterwards again against all almost alone along already also
    although always am among amongst amount an and another any anyone anything anywhere are
    around as at back be became because become becomes becoming been before beforehand being
    below besides between beyond both but by can cannot could de describe do done down due
    during each eg either else elsewhere empty enough etc even ever every everyone everything
    everywhere except few for former formerly found from full further get give go had has
    have he hence her here hers herself him himself his how i if in into is it its itself
    keep last latter least less made many may me meanwhile might mine more moreover most
    mostly move much must my myself name neither never no nor not nothing now of off often
    on once one only onto or other otherwise our ours ourselves out over own per perhaps put
    rather same see seem seemed seeming seems serious several she should show side since so
    some someone something still such take than that the their theirs them themselves then
    there thereby therefore these they this those though through throughout to together too
    toward towards un under until up upon us very via was we well were what whatever when
    where whereas whether which while who whoever whole whom whose why will with within
    without would yet you your yours yourself yourselves
    """.split()
)

HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
BBCODE_TAG_RE = re.compile(r"\[/?(?:b|i|u|s|center|spoiler|url(?:=[^\]]+)?)\]", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")
DESCRIPTOR_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
COMMON_NOUN_POS_TAGS = {"NN", "NNS"}
PROPER_NOUN_POS_TAGS = {"NNP", "NNPS"}
ADJECTIVE_POS_TAGS = {"JJ", "JJR", "JJS"}
GERUND_POS_TAGS = {"VBG"}
PRONOUN_POS_TAGS = {"PRP", "PRP$", "WP", "WP$"}
FINITE_VERB_POS_TAGS = {"VB", "VBD", "VBN", "VBP", "VBZ"}
ALLOWED_DESCRIPTOR_POS_TAGS = COMMON_NOUN_POS_TAGS | ADJECTIVE_POS_TAGS | GERUND_POS_TAGS
PRONOUN_TOKENS = {
    "he", "her", "hers", "herself", "him", "himself", "his", "i", "it", "its", "itself",
    "me", "my", "myself", "our", "ours", "ourselves", "she", "their", "theirs", "them",
    "themselves", "they", "us", "we", "who", "whom", "whose", "you", "your", "yours",
    "yourself", "yourselves",
}
CONTRACTION_RESIDUE_TOKENS = {
    "aren", "cant", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn",
    "ll", "re", "shouldn", "ve", "wasn", "weren", "won", "wouldn",
}
DESCRIPTOR_STOPWORDS = set(DEFAULT_STOPWORDS) | PRONOUN_TOKENS | CONTRACTION_RESIDUE_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit global POS-gated character-description NMF topics.")
    parser.add_argument("--characters-input", type=Path, required=True)
    parser.add_argument("--descriptions-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/k96_pos_descriptors"))
    parser.add_argument("--topic-counts", default="96")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.45)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--top-terms", type=int, default=18)
    parser.add_argument("--top-characters", type=int, default=10)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_description(value: str, *, lowercase: bool = True) -> str:
    text = html.unescape(value or "")
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = BBCODE_TAG_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    return text.lower() if lowercase else text


def import_nltk():
    try:
        import nltk
    except ImportError as error:
        raise RuntimeError("Install nltk, then run nltk.downloader averaged_perceptron_tagger_eng") from error
    nltk.pos_tag(["student"])
    return nltk


def descriptor_tokens(value: str) -> list[str]:
    return [
        token
        for token in DESCRIPTOR_TOKEN_RE.findall(normalize_description(value, lowercase=False))
        if len(token) > 1 and not token.isdigit()
    ]


def allowed_descriptor_sequence(tagged_tokens: list[tuple[str, str]]) -> bool:
    if not tagged_tokens:
        return False
    tokens = [token.lower() for token, _ in tagged_tokens]
    if any(token in DESCRIPTOR_STOPWORDS for token in tokens):
        return False
    tags = [tag for _, tag in tagged_tokens]
    if any(tag in PRONOUN_POS_TAGS for tag in tags):
        return False
    if any(tag in FINITE_VERB_POS_TAGS for tag in tags):
        return False
    if any(tag in PROPER_NOUN_POS_TAGS for tag in tags):
        return False
    if any(tag not in ALLOWED_DESCRIPTOR_POS_TAGS for tag in tags):
        return False
    if len(tags) == 1:
        return tags[0] in COMMON_NOUN_POS_TAGS | ADJECTIVE_POS_TAGS
    if "VBG" in tags:
        return any(tag in COMMON_NOUN_POS_TAGS for tag in tags)
    return all(tag in COMMON_NOUN_POS_TAGS | ADJECTIVE_POS_TAGS for tag in tags)


def descriptor_analyzer_factory(nltk: Any, ngram_min: int, ngram_max: int):
    def analyzer(value: str) -> list[str]:
        tagged = nltk.pos_tag(descriptor_tokens(value))
        candidates: list[tuple[str, ...]] = []
        for size in range(ngram_min, ngram_max + 1):
            for index in range(0, len(tagged) - size + 1):
                window = tagged[index : index + size]
                if allowed_descriptor_sequence(window):
                    candidates.append(tuple(token.lower() for token, _ in window))

        longer_candidates = [candidate for candidate in candidates if len(candidate) > 1]
        output: list[str] = []
        for candidate in candidates:
            candidate_size = len(candidate)
            if any(
                candidate_size < len(longer)
                and any(candidate == longer[start : start + candidate_size] for start in range(0, len(longer) - candidate_size + 1))
                for longer in longer_candidates
            ):
                continue
            output.append(" ".join(candidate))
        return output

    return analyzer


def character_name(character: dict) -> str:
    value = character.get("name")
    if isinstance(value, dict):
        return str(value.get("full") or value.get("userPreferred") or value.get("romaji") or "")
    return str(value or "")


def media_title(media: object) -> str:
    if not isinstance(media, dict):
        return ""
    title = media.get("title")
    if isinstance(title, dict):
        return str(title.get("english") or title.get("romaji") or title.get("native") or "")
    return str(title or "")


def media_year(media: object) -> int | None:
    if not isinstance(media, dict):
        return None
    value = media.get("startDate") or media.get("start_date") or {}
    if isinstance(value, dict) and value.get("year"):
        return int(value["year"])
    for key in ("seasonYear", "season_year", "year"):
        if media.get(key):
            return int(media[key])
    return None


def description_for_character(character: dict, descriptions: dict) -> str:
    key = str(character["id"])
    value = descriptions.get(key, "")
    if isinstance(value, dict):
        return str(value.get("description") or value.get("description_plain") or value.get("description_text") or "")
    return str(value or "")


def description_lookup(descriptions_payload: dict) -> dict[str, object]:
    if isinstance(descriptions_payload.get("characters"), list):
        return {
            str(row["id"]): row.get("description") or row.get("description_plain") or row.get("description_text") or ""
            for row in descriptions_payload["characters"]
            if isinstance(row, dict) and row.get("id") is not None
        }
    return descriptions_payload


def build_corpus(characters_payload: dict, descriptions: dict) -> tuple[list[dict], list[str], dict]:
    rows: list[dict] = []
    corpus: list[str] = []
    description_by_id = description_lookup(descriptions)
    characters = characters_payload.get("characters", [])
    characters_with_description = 0
    for character in characters:
        description = description_for_character(character, description_by_id).strip()
        if description:
            characters_with_description += 1
        if len(normalize_description(description).split()) < 5:
            continue
        first_anime = character.get("window_first_anime") or character.get("first_anime") or {}
        normalized_description = normalize_description(description)
        rows.append(
            {
                "character_id": int(character["id"]),
                "name": character_name(character),
                "native_name": character.get("native_name") or "",
                "favourites": int(character.get("favourites") or 0),
                "site_url": character.get("site_url") or f"https://anilist.co/character/{character['id']}",
                "image": character.get("image") or "",
                "first_anime": media_title(first_anime),
                "first_anime_year": media_year(first_anime),
                "description_chars": len(description),
                "description_words": len(normalized_description.split()),
            }
        )
        corpus.append(description)
    return rows, corpus, {
        "characters_input_count": len(characters),
        "characters_with_description": characters_with_description,
        "corpus_documents": len(corpus),
    }


def top_component_terms(component: np.ndarray, vocab: np.ndarray, limit: int) -> list[dict]:
    return [
        {"ngram": str(vocab[index]), "weight": round(float(component[index]), 6)}
        for index in np.argsort(component)[::-1][:limit]
        if float(component[index]) > 0
    ]


def top_topic_characters(weights: np.ndarray, rows: list[dict], topic_index: int, limit: int) -> list[dict]:
    output = []
    for index in np.argsort(weights[:, topic_index])[::-1][:limit]:
        value = float(weights[index, topic_index])
        if value <= 0:
            continue
        row = rows[index]
        output.append({**row, "topic_weight": round(value, 6)})
    return output


def fit_topic_model(k: int, tfidf, vocab: np.ndarray, rows: list[dict], args: argparse.Namespace):
    nmf = NMF(n_components=k, init="nndsvda", random_state=args.random_state, max_iter=args.max_iter)
    weights = nmf.fit_transform(tfidf).astype(np.float32)
    components = nmf.components_.astype(np.float32)
    row_sums = weights.sum(axis=1, keepdims=True)
    proportions = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0).astype(np.float32)
    relative_error = float(nmf.reconstruction_err_) / max(math.sqrt(float(tfidf.multiply(tfidf).sum())), 1e-12)
    topics = [
        {
            "topic_index": topic_index,
            "top_terms": top_component_terms(component, vocab, args.top_terms),
            "top_characters": top_topic_characters(weights, rows, topic_index, args.top_characters),
        }
        for topic_index, component in enumerate(components)
    ]
    return {
        "k": k,
        "reconstruction_error": round(float(nmf.reconstruction_err_), 6),
        "relative_reconstruction_error": round(relative_error, 6),
        "n_iter": int(nmf.n_iter_),
        "topics": topics,
    }, weights, proportions, components


def main() -> None:
    args = parse_args()
    topic_counts = [int(part.strip()) for part in args.topic_counts.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, corpus, corpus_stats = build_corpus(read_json(args.characters_input), read_json(args.descriptions_input))
    nltk = import_nltk()
    vectorizer = TfidfVectorizer(
        analyzer=descriptor_analyzer_factory(nltk, args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(corpus)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    write_json(args.output_dir / "character_rows.json", rows)
    write_json(args.output_dir / "tfidf_vocabulary.json", [{"index": int(i), "ngram": str(term), "idf": round(float(vectorizer.idf_[i]), 6)} for i, term in enumerate(vocab)])
    sweep = {
        "generated_at": utc_now(),
        "source": "fit_topics.py",
        "parameters": {
            "topic_counts": topic_counts,
            "feature_mode": "pos_gated_descriptor_ngrams",
            "proper_nouns": "excluded",
            "standalone_gerunds": "excluded",
            "subgram_suppression": "document-level longest-match; lower-order n-grams contained in any valid longer n-gram are omitted for that character",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "random_state": args.random_state,
        },
        "tfidf": {**corpus_stats, "matrix_shape": [int(tfidf.shape[0]), int(tfidf.shape[1])], "nonzero_entries": int(tfidf.nnz)},
        "fits": [],
    }
    for k in topic_counts:
        print(f"fitting NMF k={k} on TF-IDF {tfidf.shape[0]} x {tfidf.shape[1]}", flush=True)
        summary, weights, proportions, components = fit_topic_model(k, tfidf, vocab, rows, args)
        matrix_name = f"nmf_k{k:03d}_matrices.npz"
        np.savez_compressed(args.output_dir / matrix_name, character_topic_weights=weights, character_topic_proportions=proportions, topic_ngram_components=components, character_ids=np.asarray([row["character_id"] for row in rows], dtype=np.int64))
        summary["matrix_file"] = matrix_name
        write_json(args.output_dir / f"nmf_k{k:03d}_summary.json", summary)
        sweep["fits"].append({"k": k, "summary_file": f"nmf_k{k:03d}_summary.json", "matrix_file": matrix_name, "relative_reconstruction_error": summary["relative_reconstruction_error"]})
    write_json(args.output_dir / "global_nmf_topic_sweep.json", sweep)
    print(f"wrote {args.output_dir / 'global_nmf_topic_sweep.json'}")


if __name__ == "__main__":
    main()
