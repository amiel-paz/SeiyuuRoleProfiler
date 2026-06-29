# AniDB Character Tag Cache

This project can enrich the local AniList seiyuu-role corpus with AniDB's
anime XML and cast crosswalk data. AniDB's page-level character tags, such as
`personality: tsundere`, `role: student council president`, and
`abilities: strategist`, are not fetched by this pipeline because AniDB policy
disallows scraping HTML pages.

The cache runner is:

```sh
.venv/bin/python scripts/cache_anidb_character_tags.py
```

It writes:

- `data/external/anidb/anime-list-full.json`
- `data/external/anidb/anime_xml/<aid>.xml`
- `data/external/anidb/anidb_character_crosswalk.json`
- `data/external/anidb/anidb_character_tag_cache.json`

## Provenance

The runner uses the community `Fribb/anime-lists` mapping to connect local
AniList/MAL anime IDs to AniDB anime IDs.

AniDB's public HTTP API is intentionally limited and anime-centric. The anime
XML includes cast rows with AniDB character IDs and seiyuu rows, which is enough
to build a reproducible character crosswalk for our local corpus. AniDB's public
API documentation does not expose a character-tag endpoint. AniDB's API page also
asks developers not to parse HTML directly when the available APIs/resources do
not cover a use case.

Because of that, the runner only fetches API/mirror-backed anime XML. It may
parse local character HTML files if they are provided from a permitted source,
but it does not crawl AniDB character pages.

## Anime XML Setup

By default the runner uses Kometa's public AniDB mirror for cached anime XML:

```sh
.venv/bin/python scripts/cache_anidb_character_tags.py \
  --fetch-anime-xml \
  --anime-fetch-jobs 8
```

This avoids requiring AniDB credentials for the anime/cast crosswalk layer. If
you prefer the official AniDB HTTP API directly, register an AniDB HTTP API
client in your AniDB account, then provide the client name/version:

```sh
export ANIDB_CLIENT=your_client_name
export ANIDB_CLIENTVER=1

.venv/bin/python scripts/cache_anidb_character_tags.py \
  --anime-xml-source official \
  --fetch-anime-xml
```

The runner keeps raw XML in `data/external/anidb/anime_xml/` and will not fetch
again when a file already exists.

Use `--max-anime-fetches` for incremental runs:

```sh
.venv/bin/python scripts/cache_anidb_character_tags.py \
  --fetch-anime-xml \
  --anime-fetch-jobs 4 \
  --max-anime-fetches 25
```

## Character Tags

The JSON cache exposes `character_page_values_by_category`, but this remains
empty unless local character HTML files are supplied from a source that permits
that use. Rebuild the JSON cache with:

```sh
.venv/bin/python scripts/cache_anidb_character_tags.py
```

When permitted local character pages are present, the final JSON keeps each
tag's category, tag ID, name, and URL.

The final JSON also stores the global tag taxonomy found in cached AniDB anime
XML:

- `anime_tag_values_by_category`: parent tag name -> all child tag names.
- `anidb_character_candidate_values`: the child values under AniDB's legacy
  `TO BE MOVED TO CHARACTER` tag bucket.

This XML taxonomy is useful as a deterministic global vocabulary and includes
values such as `tsundere`, `hard-working`, and `sarcastic`, but it is not the
same as the richer page-level character categories such as `personality`,
`role`, or `traits`.

## Matching

The crosswalk is deterministic:

1. map local AniList/MAL anime IDs to AniDB anime IDs;
2. parse each cached AniDB anime XML cast list;
3. match local characters to AniDB characters within the same anime by normalized
   character name, including surname/given-name reversal;
4. prefer matches where the AniDB seiyuu row also matches the local seiyuu name.

This keeps the tags tied to explicit AniDB character IDs rather than fuzzy
global name search.
