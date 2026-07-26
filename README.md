# mcp
list of my mcp tools

## Email configuration

Email tools use IMAP and SMTP settings from `.env`. For Gmail, use a Google
App Password instead of the account password. See `.env.example` for all
required variables. Never commit `.env`.

## Google Calendar configuration

Place the Google Desktop OAuth client file at `credentials.json`. The first
Calendar tool call opens browser authorization and creates `token.json`. Both
files are ignored by Git. See `.env.example` for path and timezone settings.
