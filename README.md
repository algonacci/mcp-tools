# mcp
list of my mcp tools

## Setup

Create or update `.env` with the interactive setup wizard:

```bash
python setup_env.py
```

The script works on Windows, macOS, and Linux. Each integration is optional,
existing values are preserved by default, and secret values are hidden while
typing. Run it again whenever you want to add or update an integration.

You do not need to configure every integration. The MCP server still starts
with a partial or missing `.env`; tools such as GARUDA, Wikipedia, Google News,
Open-Meteo weather, PDF, and local file readers remain available. Integrations
that need credentials are left out of the tool list until they are configured,
and the server names them on stderr at startup:

| Integration | Exposed when |
|---|---|
| Tavily (`tavily_search`, `extract_url`) | `TAVILY_API_KEY` is set |
| Exa (`exa_*`) | `EXA_API_KEY` is set |
| Email | `EMAIL_ADDRESS` and `EMAIL_SECRET` are set |
| Google Calendar and Drive | `credentials.json` or `token.json` exists |

Restart the server after adding credentials so the new tools are registered.

## Weather

Open-Meteo tools provide place search, current conditions, hourly forecasts,
and daily forecasts without an API key. Search for a place with
`search_weather_locations`, then pass its coordinates to a forecast tool.

## Currency conversion

Frankfurter tools list currencies and providers, return current, historical,
multi-currency, and provider-specific rates, and convert amounts without an API
key. Currency codes use three-letter notation such as `USD`, `IDR`, and `EUR`.
`create_exchange_rate_chart` saves a PNG under `downloads/charts` and also
returns it as MCP image content. arXiv papers are stored under `downloads/arxiv`.
`create_data_chart` provides the same high-DPI output for selected columns in
CSV or Excel files, with labels, annotations, summary statistics, and source notes.
Use `inspect_data_file` first to discover sheets, columns, data types, and sample rows.
For dummy, generated, or small datasets, `create_inline_data_chart` accepts records
directly and returns the PNG without creating a script or temporary data file.

## Exa web search

Exa tools need `EXA_API_KEY`; the free tier gives $10 of credits per month and every
response includes `cost_dollars`.

- `exa_search`: web search with highlights or full text; filters for category
  (`company`, `news`, `publication`, `people`, ...), domains, and published-date range
- `exa_get_contents`: read known URLs, optionally only the passages relevant to a query
- `exa_deep_search`: multi-step search that synthesizes a cited answer, as text or
  shaped by a JSON Schema (`output_schema`)
- `exa_research_start` / `exa_research_get`: Exa's research agent for lists and reports.
  Runs take minutes, so `start` returns a `run_id` and `get` waits up to 50 seconds per call

Results are capped at 25, URLs at 10 per call, and research effort at `medium` to keep a
single call from draining the budget. The same tools ship standalone in
[mcp-exa](https://github.com/algonacci/mcp-exa).

## GARUDA academic search

`search_garuda` mirrors the filters on GARUDA's own search form
(`https://garuda.kemdiktisaintek.go.id/documents`): pass `search_field`
(`title`, `abstract`, `author`, or `doi`) to restrict which field the query
matches against — use `author` for exact author-name lookups, since plain
keyword search does not reliably match on author names. Additional filters:
`publisher` (name, min. 3 characters), `pdf_only` (downloadable PDF only),
and `year_from`/`year_to` (publication year range).

## Crossref scholarly metadata

Crossref tools search ~170 million works from every publisher and need no API key.
Set `CROSSREF_MAILTO` to join Crossref's faster "polite" pool, and optionally
`OPENALEX_API_KEY` (free) for a larger `snowball_doi` budget; it runs keyless otherwise.

- `search_crossref`: keyword, author, citation-string (`bibliographic`), journal, funder and date/type filters
- `get_crossref_work`: full record for a DOI plus which metadata fields are missing
- `cite_dois`: APA, IEEE, Vancouver, Chicago, MLA, Harvard, any CSL style, BibTeX, RIS (arXiv DOIs too)
- `resolve_dois`: verify every DOI found in pasted text
- `get_crossref_references`, `find_related_works`, `snowball_doi`: references, similar works,
  and works citing a paper (OpenAlex) for literature-review snowballing
- `analyze_crossref_topic`, `get_crossref_author`, `get_crossref_journal`, `get_crossref_funder`:
  topic trends, researcher, journal and funder profiles

Crossref has no exact-phrase search, so plain queries match any word; `analyze_crossref_topic`
counts only titles containing the exact phrase. The same tools ship standalone in
[mcp-crossref](https://github.com/algonacci/mcp-crossref).

## PlantUML

`render_plantuml` renders source containing `@startuml` and `@enduml` as PNG
through the official PlantUML Server. Results are saved under
`downloads/plantuml` and returned as MCP image content for Telegram delivery.

## Email configuration

Email tools use IMAP and SMTP settings from `.env`. For Gmail, use a Google
App Password instead of the account password. See `.env.example` for all
required variables. Never commit `.env`.

## Google Calendar and Drive configuration

Place the Google Desktop OAuth client file at `credentials.json`, with both the
Google Calendar API and Google Drive API enabled in that Google Cloud project.
Both `credentials.json` and `token.json` are ignored by Git.

Authorization is a two-step tool call, so it works the same whether the server
runs on your laptop or on a machine with no browser at all:

1. Call `google_auth_start`. It returns an authorization URL.
2. Approve access in a browser — any browser, on any device.
3. Call `google_auth_complete` with the URL you land on.

On a desktop, steps 2 and 3 usually happen by themselves: the browser is opened
for you and the redirect is caught on `http://localhost:8765` (set
`GOOGLE_OAUTH_PORT` to change it), so approving in the browser is all you do.

On a headless server, or when you approve on a phone, that redirect cannot be
reached and the browser shows a connection error. That is expected — the
authorization code is in the address bar. Copy the whole URL
(`http://localhost:8765/?code=...`) and pass it to `google_auth_complete`. It
also accepts the bare `code` value if that is easier to copy.

Whichever path finishes first wins, so it is safe to start in the browser and
fall back to pasting. Tokens refresh themselves afterwards; you only repeat this
if the token is revoked or the requested scopes change.

Drive access uses the read-only scope. It can search files, inspect metadata,
read Docs and Sheets as text, and download or export files. Existing Calendar
users will be asked to authorize again because the shared OAuth token now has
an additional scope. See `.env.example` for path and timezone settings.
