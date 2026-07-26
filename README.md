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
Open-Meteo weather, PDF, and local file readers remain available. A tool that
needs missing credentials returns a setup error only when that tool is called.

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

## PlantUML

`render_plantuml` renders source containing `@startuml` and `@enduml` as PNG
through the official PlantUML Server. Results are saved under
`downloads/plantuml` and returned as MCP image content for Telegram delivery.

## Email configuration

Email tools use IMAP and SMTP settings from `.env`. For Gmail, use a Google
App Password instead of the account password. See `.env.example` for all
required variables. Never commit `.env`.

## Google Calendar and Drive configuration

Place the Google Desktop OAuth client file at `credentials.json`. The first
Calendar or Drive tool call opens browser authorization and creates
`token.json`. Enable both the Google Calendar API and Google Drive API in the
Google Cloud project. Both files are ignored by Git.

Drive access uses the read-only scope. It can search files, inspect metadata,
read Docs and Sheets as text, and download or export files. Existing Calendar
users will be asked to authorize again because the shared OAuth token now has
an additional scope. See `.env.example` for path and timezone settings.
