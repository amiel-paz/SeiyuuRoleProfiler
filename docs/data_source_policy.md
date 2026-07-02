# Data Source Policy

This project is intended to be hostable, so enrichment sources should be safe to
use outside a local prototype. API access alone is not sufficient: the content
license and platform terms still control reuse.

## Current Rule

- Do not use Fandom-derived extracts or tags in the public dataset.
- Keep raw third-party prose out of git.
- Keep per-character provenance for every external source used to produce tags.
- Prefer structured/open datasets over prose scraping.

## Preferred Sources

### Wikidata

Use for structured identifiers and any personality-trait statements that exist.
Wikidata main/entity data is CC0, which is the cleanest fit for a public
profiler.

### Wikimedia / Wikipedia APIs

Usable with attribution and CC BY-SA compliance. Text retrieved through the API
inherits the wiki page license. This is acceptable for compact, attributed
source-backed enrichment, but raw prose should not be redistributed.

### VNDB

Useful for character traits and visual-novel role metadata. The API is free for
non-commercial use and the data is subject to VNDB's data license. Treat VNDB as
allowed for a non-commercial hosted profiler only with attribution and
share-alike/open-database compliance.

### Bangumi

Promising Chinese-language source for subject and character metadata. Bangumi
explicitly allows developers to build apps and services with its API and archive
data, but platform-data redistribution restrictions mean we should use it with
care: attribution, minimal fields, no raw dumps in git, and no resale or bulk
republication of platform data.

### Independent CC BY-SA MediaWiki Sites

Some non-Fandom fan wikis, such as JoJo's Bizarre Encyclopedia, publish text
under CC BY-SA and expose MediaWiki APIs. These can be used on a narrow,
source-by-source allowlist after verifying each site's license and API etiquette.

## Avoid For Public Builds

- Fandom API/wiki text: useful but not clean enough for this app because of
  automated retrieval and software/AI-use restrictions in platform terms.
- Moegirlpedia Chinese text: CC BY-NC-SA / noncommercial; attractive coverage,
  but the noncommercial restriction is too limiting for a future-proof hosted
  profiler.
- Unofficial scraper APIs such as Jikan for MAL: useful as references, but they
  scrape upstream sites and push terms risk onto the user.
- Kaggle/third-party anime character dumps unless their provenance and license
  are explicit and compatible.

## Public-Build Recommendation

The public enrichment stack should be:

1. Existing AniList/MAL-style character descriptions already used by the app,
   with source links and favorites dates.
2. Wikidata CC0 structured fields.
3. VNDB character traits where matched, with data-license attribution.
4. Optional Bangumi character/subject metadata after a small terms-compliant
   matching pass.
5. Optional allowlisted independent CC BY-SA MediaWiki APIs, only after verifying
   each site's license page.

Every generated descriptor should retain source provenance, even if the UI only
shows it under a quiet "Sources" or "Attribution" control.
