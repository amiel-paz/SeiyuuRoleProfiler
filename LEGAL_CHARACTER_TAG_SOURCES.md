# Legal Character Tag Sources

We should not crawl AniDB character HTML. AniDB's public policy disallows HTML
scraping, and the available API/mirror-backed XML only exposes anime metadata,
cast IDs, and anime tag trees. That XML remains useful for deterministic
crosswalks, but not for page-level character traits.

## Recommended Source: VNDB

VNDB's HTTPS API is the best legal source found so far for concrete character
traits. It is official, documented, non-commercial use is allowed, and the
`/character` endpoint can return trait rows attached to characters. Traits carry
top-level groups such as `Personality`, `Role`, `Body`, `Hair`, `Eyes`,
`Clothes`, `Engages in`, and `Subject of`.

Runner:

```sh
.venv/bin/python scripts/cache_vndb_character_traits.py
```

The runner:

- searches each local AniList role character by name;
- caches the raw VNDB search response per local AniList character ID in
  `data/external/vndb/raw_character_search/`;
- only accepts exact, alias, or reordered-name matches;
- rejects one-token matches unless the local anime title overlaps the VNDB VN
  title tokens;
- excludes spoiler traits above `--max-spoiler 0` by default;
- excludes sexual traits by default;
- writes `data/external/vndb/vndb_character_traits.json`;
- stores `values_by_group` as a reusable group -> trait-values dictionary.

This is conservative by design. It should prefer missing coverage over false
cross-database matches.

Run the full current corpus with:

```sh
.venv/bin/python scripts/cache_vndb_character_traits.py \
  --limit 0 \
  --sleep-seconds 1.6 \
  --checkpoint-every 50
```

Rebuild the derived JSON from already-cached raw responses without network:

```sh
.venv/bin/python scripts/cache_vndb_character_traits.py \
  --limit 0 \
  --offline \
  --sleep-seconds 0
```

Use `--refresh-raw` only when intentionally re-querying VNDB.

Known useful examples from the API:

- Nino Nakano: `Personality: Classic Tsundere`, `Personality: Outgoing`,
  `Role: High School Student`.
- Kirino Kousaka: `Personality: Modern Tsundere`, `Personality: Otaku`,
  `Role: Popular`.
- Kotori Itsuka: `Personality: Modern Tsundere`, `Role: Commander`,
  `Role: Non-blood-related Sister`.

References:

- https://api.vndb.org/kana
- https://vndb.org/d6

## Secondary Candidate: Anime Characters Database

Anime Characters Database exposes API tooling and a public character tag list.
It may be useful for image/appearance-heavy tags and some role labels, but the
public API surface appears less directly suited to personality/archetype
profiling than VNDB. Treat it as a second-pass enrichment source after testing
coverage and terms carefully.

References:

- https://www.animecharactersdatabase.com/api.php
- https://www.animecharactersdatabase.com/taglist.php

## Description APIs

AniList, Jikan/MAL, and Kitsu are still useful for descriptions, images, role
credits, and URLs. They do not appear to provide a broad concrete character
trait ontology comparable to VNDB traits.

References:

- https://jikan.moe/
- https://hummingbird-me.github.io/api-docs/
