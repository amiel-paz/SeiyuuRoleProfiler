# LLM Character Tag Extraction

This is a deterministic-ish fallback for characters without structured VNDB
traits. It asks a local Ollama model to extract tags from the supplied
MAL/AniList description only, then caches the raw model response so the derived
JSON can be rebuilt.

Runner:

```sh
.venv/bin/python scripts/cache_llm_character_tags.py \
  --descriptions-input path/to/descriptions.json
```

The extraction schema is intentionally many-to-many:

```json
{
  "role": [
    {"tag": "student council president", "evidence": "exact span", "confidence": "high"}
  ],
  "personality": [
    {"tag": "sharp-tongued", "evidence": "exact span", "confidence": "medium"}
  ],
  "traits": [
    {"tag": "skilled fighter", "evidence": "exact span", "confidence": "high"}
  ]
}
```

Each category is an array and may contain zero, one, or many tags. The runner
keeps only tags whose evidence is an exact span from the description. This is
the guardrail that prevents the model from using fandom memory or anime-title
priors.

By default the runner sends `think: false` to Ollama. This keeps extraction
fast and schema-focused. Thinking can be enabled explicitly for experiments with
`--think true`, `--think low`, `--think medium`, `--think high`, or
`--think max`, but those runs should be cached separately because latency and
outputs may differ.

Outputs:

- `data/external/llm/raw_character_tags/<anilist_character_id>.json`: raw
  prompt, model options, and Ollama response.
- `data/external/llm/character_description_tags.json`: validated tags,
  aggregate counts, and category -> tag-value dictionaries.

Recommended probe:

```sh
.venv/bin/python scripts/cache_llm_character_tags.py \
  --descriptions-input path/to/descriptions.json \
  --name "Nino Nakano" \
  --name "Kirino Kousaka" \
  --limit 0 \
  --refresh-raw
```

Offline rebuild from raw model responses:

```sh
.venv/bin/python scripts/cache_llm_character_tags.py \
  --descriptions-input path/to/descriptions.json \
  --limit 0 \
  --offline \
  --sleep-seconds 0
```

For reproducibility, keep the model name, temperature, seed, thinking setting,
prompt, and raw response cache with any derived analysis.
