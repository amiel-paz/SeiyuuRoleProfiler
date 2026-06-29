#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
BBCODE_TAG_RE = re.compile(r"\[/?(?:b|i|u|s|center|spoiler|url(?:=[^\]]+)?)\]", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z]*(?:-[A-Za-z][A-Za-z]*)?")

STOPWORDS = frozenset(
    """
    a about above across after again against all almost alone along already also although
    always am among an and another any anyone anything anywhere are around as at back be
    because become been before being below between both but by can cannot could de do done
    down due during each either else enough etc even ever every everyone everything except
    few for former from full further get give go had has have he her here hers herself him
    himself his how i if in into is it its itself keep last least less made many may me
    might more most mostly much must my myself name neither never no nor not nothing now of
    off often on once one only or other our out over own perhaps put rather same see seem
    several she should since so some someone something still such take than that the their
    them themselves then there these they this those though through to together too under
    until up upon us very via was we well were what when where whether which while who whom
    whose why will with within without would yet you your yourself
    """.split()
)
CONTRACTION_RESIDUES = {
    "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn",
    "ll", "re", "shouldn", "ve", "wasn", "weren", "won", "wouldn",
}

NOUN_TAGS = {"NN", "NNS"}
PERSON_NOUN_TAGS = {"NN", "NNS"}
MODIFIER_TAGS = {"JJ", "JJR", "JJS", "VBG", "VBN", "NN", "NNS"}
CHUNK_TAGS = MODIFIER_TAGS
PROPER_TAGS = {"NNP", "NNPS"}
BAD_LEXNAMES = {
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.food",
    "noun.location",
    "noun.object",
    "noun.plant",
    "noun.quantity",
    "noun.shape",
    "noun.substance",
    "noun.time",
}
GENERIC_ATTRIBUTE_LEXNAMES = {"noun.attribute", "noun.cognition", "noun.feeling", "noun.Tops"}
ROLE_LEXNAMES = {"noun.person", "noun.group"}
EVENT_LEXNAMES = {"noun.act", "noun.communication", "noun.relation", "noun.state"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache deterministic WordNet-filtered semantic descriptor features.")
    parser.add_argument(
        "--characters-input",
        type=Path,
        default=Path("data/external/anilist-top-characters-2007-2026-min100.json"),
    )
    parser.add_argument(
        "--descriptions-input",
        type=Path,
        default=Path("data/external/anilist-top-characters-media-appearances-2007-2026-min100-descriptions.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/semantic_wordnet_descriptors"))
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.45)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--max-chunk-size", type=int, default=5)
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


def normalize_text(value: str, *, lowercase: bool = True) -> str:
    text = html.unescape(value or "")
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = BBCODE_TAG_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[_*`]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def import_nltk_wordnet():
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError as error:
        raise RuntimeError("Install nltk before running semantic descriptor extraction.") from error
    try:
        nltk.pos_tag(["outgoing", "attitude"])
        wn.synsets("personality")
    except LookupError as error:
        raise RuntimeError(
            "Missing NLTK data. Run: python -m nltk.downloader averaged_perceptron_tagger_eng wordnet omw-1.4"
        ) from error
    return nltk, wn


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


def description_lookup(descriptions_payload: dict) -> dict[str, object]:
    if isinstance(descriptions_payload.get("characters"), list):
        return {
            str(row["id"]): row.get("description") or row.get("description_plain") or row.get("description_text") or ""
            for row in descriptions_payload["characters"]
            if isinstance(row, dict) and row.get("id") is not None
        }
    return descriptions_payload


def build_corpus(characters_payload: dict, descriptions_payload: dict) -> tuple[list[dict], list[str], dict]:
    rows: list[dict] = []
    corpus: list[str] = []
    description_by_id = description_lookup(descriptions_payload)
    characters = characters_payload.get("characters", [])
    characters_with_description = 0
    for character in characters:
        description = str(description_by_id.get(str(character["id"]), "") or "").strip()
        if description:
            characters_with_description += 1
        if len(normalize_text(description).split()) < 5:
            continue
        first_anime = character.get("window_first_anime") or character.get("first_anime") or {}
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
                "description_words": len(normalize_text(description).split()),
            }
        )
        corpus.append(description)
    return rows, corpus, {
        "characters_input_count": len(characters),
        "characters_with_description": characters_with_description,
        "corpus_documents": len(corpus),
    }


def lemma_token(token: str, tag: str, wn: Any) -> str:
    token = token.lower()
    token = token[:-5] if token.endswith("-like") else token
    if tag.startswith("J"):
        return wn.morphy(token, wn.ADJ) or token
    if tag.startswith("V"):
        if tag in {"VBG", "VBN"} and wn.synsets(token.replace("-", "_"), pos=wn.ADJ):
            return token
        return wn.morphy(token, wn.VERB) or token
    return wn.morphy(token, wn.NOUN) or token


def synsets_for_phrase(phrase: str, wn: Any) -> list[Any]:
    key = phrase.replace(" ", "_").replace("-", "_")
    return wn.synsets(key)


def wordnet_pos_for_tag(tag: str, wn: Any) -> str | None:
    if tag.startswith("JJ"):
        return wn.ADJ
    if tag.startswith("NN"):
        return wn.NOUN
    if tag.startswith("VB"):
        return wn.ADJ if tag in {"VBG", "VBN"} else wn.VERB
    return None


def first_usable_synset(phrase: str, wn: Any, pos: str | None = None) -> Any | None:
    key = phrase.replace(" ", "_").replace("-", "_")
    synsets = wn.synsets(key, pos=pos) if pos else wn.synsets(key)
    for synset in synsets:
        if synset.lexname() not in BAD_LEXNAMES:
            return synset
    return None


def head_synset(head: str, wn: Any) -> Any | None:
    synsets = wn.synsets(head, pos=wn.NOUN)
    if not synsets:
        return None
    primary = synsets[0]
    return None if primary.lexname() in BAD_LEXNAMES else primary


def clean_feature(value: str) -> str:
    value = value.replace("-like", "")
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def span_candidates(tagged: list[tuple[str, str]], max_chunk_size: int) -> list[tuple[list[tuple[str, str]], str]]:
    output: list[tuple[list[tuple[str, str]], str]] = []
    n = len(tagged)
    for start in range(n):
        for end in range(start + 1, min(n, start + max_chunk_size) + 1):
            window = tagged[start:end]
            tokens = [token.lower() for token, _ in window]
            tags = [tag for _, tag in window]
            if any(len(token) < 2 or token in STOPWORDS or token in CONTRACTION_RESIDUES for token in tokens):
                continue
            if any(tag in PROPER_TAGS for tag in tags):
                continue
            if any(tag not in CHUNK_TAGS for tag in tags):
                continue
            if len(window) == 1 and not (tags[0].startswith("JJ") or tags[0] == "VBG" or tokens[0].endswith("-like")):
                continue
            if len(window) > 1 and tags[-1] not in NOUN_TAGS:
                continue
            output.append((window, "pos_span"))
    return output


def semantic_features_for_span(window: list[tuple[str, str]], wn: Any) -> list[dict]:
    lemmas = [lemma_token(token, tag, wn) for token, tag in window]
    phrase = clean_feature(" ".join(lemmas))
    if not phrase:
        return []

    features: list[dict] = []
    exact_pos = wordnet_pos_for_tag(window[-1][1], wn) if len(window) == 1 else wn.NOUN
    exact = first_usable_synset(phrase, wn, exact_pos)
    if exact is not None:
        if not exact.lexname().startswith("verb."):
            features.append(
                {
                    "feature": phrase,
                    "kind": "surface_wordnet_phrase",
                    "lexname": exact.lexname(),
                    "synset": exact.name(),
                }
            )

    if len(lemmas) == 1:
        token, tag = window[0]
        if token.lower().endswith("-like"):
            base = clean_feature(lemmas[0])
            if base:
                features.append({"feature": base, "kind": "morphological_like", "lexname": "", "synset": ""})
            return features
        pos = wordnet_pos_for_tag(tag, wn)
        synsets = wn.synsets(phrase.replace(" ", "_"), pos=pos) if pos else synsets_for_phrase(phrase, wn)
        if tag.startswith("JJ") or tag == "VBG":
            if any(synset.lexname().startswith("adj.") for synset in synsets) or not synsets:
                features.append({"feature": phrase, "kind": "surface_modifier", "lexname": "adj_or_unknown", "synset": synsets[0].name() if synsets else ""})
        return features

    head = lemmas[-1]
    head_syn = head_synset(head, wn)
    if head_syn is None:
        return features
    head_lexname = head_syn.lexname()
    modifiers = clean_feature(" ".join(lemmas[:-1]))
    if head_lexname in ROLE_LEXNAMES:
        features.append(
            {
                "feature": phrase,
                "kind": "surface_role_phrase",
                "lexname": head_lexname,
                "synset": head_syn.name(),
            }
        )
        features.append(
            {
                "feature": head_syn.lemmas()[0].name().replace("_", " "),
                "kind": "wordnet_role_head",
                "lexname": head_lexname,
                "synset": head_syn.name(),
            }
        )
    elif head_lexname in GENERIC_ATTRIBUTE_LEXNAMES and modifiers:
        modifier_tags = [tag for _, tag in window[:-1]]
        if any(tag.startswith("JJ") or tag in {"VBG", "VBN"} for tag in modifier_tags):
            features.append(
                {
                    "feature": modifiers,
                    "kind": "attribute_modifier",
                    "lexname": head_lexname,
                    "synset": head_syn.name(),
                }
            )
        else:
            features.append(
                {
                    "feature": phrase,
                    "kind": "surface_attribute_phrase",
                    "lexname": head_lexname,
                    "synset": head_syn.name(),
                }
            )
    elif head_lexname in EVENT_LEXNAMES:
        features.append(
            {
                "feature": phrase,
                "kind": "surface_event_phrase",
                "lexname": head_lexname,
                "synset": head_syn.name(),
            }
        )
    return features


def extract_semantic_features(value: str, nltk: Any, wn: Any, max_chunk_size: int) -> list[dict]:
    text = normalize_text(value, lowercase=False)
    collected: dict[str, dict] = {}
    for sentence in SENTENCE_SPLIT_RE.split(text):
        tokens = TOKEN_RE.findall(sentence)
        if not tokens:
            continue
        tagged = nltk.pos_tag(tokens)
        for window, rule in span_candidates(tagged, max_chunk_size):
            source_phrase = " ".join(token for token, _ in window)
            for feature in semantic_features_for_span(window, wn):
                term = clean_feature(feature["feature"])
                if len(term) < 3 or term in STOPWORDS:
                    continue
                payload = {
                    "feature": term,
                    "source_phrase": source_phrase,
                    "rule": rule,
                    "kind": feature["kind"],
                    "lexname": feature["lexname"],
                    "synset": feature["synset"],
                }
                collected.setdefault(term, payload)
    return sorted(collected.values(), key=lambda row: (row["feature"], row["source_phrase"]))


def main() -> None:
    args = parse_args()
    nltk, wn = import_nltk_wordnet()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, corpus, corpus_stats = build_corpus(read_json(args.characters_input), read_json(args.descriptions_input))

    feature_docs: list[list[str]] = []
    feature_details_by_character: list[dict] = []
    kind_counts: Counter[str] = Counter()
    lexname_counts: Counter[str] = Counter()
    for row, description in zip(rows, corpus):
        features = extract_semantic_features(description, nltk, wn, args.max_chunk_size)
        feature_docs.append([feature["feature"] for feature in features])
        for feature in features:
            kind_counts[feature["kind"]] += 1
            if feature["lexname"]:
                lexname_counts[feature["lexname"]] += 1
        feature_details_by_character.append(
            {
                "character_id": row["character_id"],
                "name": row["name"],
                "feature_count": len(features),
                "features": features,
            }
        )

    vectorizer = TfidfVectorizer(
        analyzer=lambda value: value,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(feature_docs).astype(np.float32)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    idf = vectorizer.idf_

    sparse.save_npz(args.output_dir / "character_semantic_tfidf.npz", matrix)
    write_json(args.output_dir / "character_rows.json", rows)
    write_json(args.output_dir / "semantic_vocabulary.json", [{"index": int(i), "feature": str(term), "idf": round(float(idf[i]), 6)} for i, term in enumerate(vocab)])
    write_json(args.output_dir / "semantic_features_by_character.json", feature_details_by_character)
    write_json(
        args.output_dir / "semantic_matrix_metadata.json",
        {
            "generated_at": utc_now(),
            "source": "cache_semantic_descriptor_matrix.py",
            "parameters": {
                "min_df": args.min_df,
                "max_df": args.max_df,
                "max_features": args.max_features,
                "max_chunk_size": args.max_chunk_size,
                "semantic_resource": "NLTK WordNet lexnames and synsets",
                "bad_lexnames": sorted(BAD_LEXNAMES),
                "feature_policy": "deterministic POS span candidates filtered/canonicalized by WordNet phrase/head lexnames; role heads are added as WordNet canonical semantic features",
            },
            "corpus": corpus_stats,
            "semantic_features": {
                "raw_feature_assignments": int(sum(len(doc) for doc in feature_docs)),
                "characters_with_features": int(sum(bool(doc) for doc in feature_docs)),
                "kind_counts": dict(sorted(kind_counts.items())),
                "lexname_counts": dict(sorted(lexname_counts.items())),
            },
            "tfidf": {
                "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "nonzero_entries": int(matrix.nnz),
            },
        },
    )
    print(f"wrote {args.output_dir / 'character_semantic_tfidf.npz'}")
    print(f"wrote {args.output_dir / 'semantic_features_by_character.json'}")
    print(f"semantic matrix {matrix.shape[0]} x {matrix.shape[1]} nnz={matrix.nnz}")


if __name__ == "__main__":
    main()
