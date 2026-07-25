# MCP Tool Wishlist

Ideas for new tools to add to this server, mainly driven by what would make Kumo (the personal
Telegram agent gateway at github.com/algonacci/kumo) more useful as a day-to-day assistant. Kumo's
own design principle: anything that is a genuine *capability* (search, read a data source, talk to
an external service) belongs here as an MCP tool, not as native Kumo code. Kumo stays native only
for things that touch the gateway itself — receiving Telegram messages/photos, running host
commands, scheduling, memory. See kumo's ROADMAP.md (Phase 4c) for that split in more detail.

Already covered by `server.py` — do not duplicate: database access (`connect_database`,
`execute_query`, `list_tables`, `describe_table`, `disconnect_database`), news (`search_news`,
`get_top_news`, `get_topic_news`), web search (`tavily_search`, `search`, `extract_url`), PDF/Excel/
notebook reading (`read_pdf`, `read_excel`, `read_notebook`), Wikipedia (`summary`, `page`,
`random`, `set_lang`), academic papers (`search_papers`, `download_paper`, `search_garuda`,
`get_garuda_detail`, `search_ieee`, `search_sciencedirect`), and crypto/market data (`get_price`,
`get_coin_detail`, `get_top_coins`, `search_coin`, `get_global_market`, `get_price_history`,
`compare_coins`).

## High priority — personal assistant essentials

- [ ] **Gmail** — read recent messages, search by sender/subject/label, send a draft or reply. This
  was the original motivation for this list: reading email is a clear MCP capability, not
  something Kumo should ever implement natively. A read-only pass (list/search/read) is safer to
  ship first; sending mail should probably require the same kind of explicit confirmation Kumo
  already asks for on `run_command`, since a Kumo tool call maps 1:1 to an MCP call with no
  in-between review by this server.
- [ ] **Google Calendar** — list upcoming events, check free/busy for a time range, create an
  event. Read-only listing first; creating/deleting events is the side-effecting half and should
  be treated with the same care as sending email.
- [ ] **Google Drive / Docs** — search files by name, read a Doc's or Sheet's content as text.
  Useful once Gmail/Calendar exist, since a lot of personal workflows touch all three together.

## Medium priority — clearly useful, less urgent

- [ ] **Browser automation** — `playwright` and `playwright-stealth` are already dependencies in
  `pyproject.toml` but no `@mcp.tool()` currently exposes them. If that was work in progress,
  finishing it would unlock "open this page and tell me X" for sites `extract_url` can't handle
  (JS-rendered pages, login walls, interactive steps). Both OpenClaw and Hermes list `browser` as a
  first-class capability.
- [ ] **Local filesystem search** — full-text or filename search across a configured directory
  tree, complementing Kumo's own `read_file`/`list_directory` (which only lists/reads a path you
  already know) with "find the file that mentions X."
- [ ] **Weather** — current conditions and short forecast by city/coordinates. Small, self-
  contained, commonly requested by a personal-assistant bot.
- [ ] **Currency/unit conversion** — a lot of "how much is X in Y" questions currently have no
  clean tool; `get_price`/`get_price_history` only cover crypto.
- [ ] **Calendar-adjacent reminders vs. Kumo's own `schedule_task`** — worth explicitly *not*
  building a duplicate scheduling tool here. Kumo's `schedule_task` already covers "run this prompt
  later"; an MCP calendar tool should stay a read/write interface to Google Calendar itself, not a
  second scheduler.

## Lower priority — nice to have

- [ ] **Notion / Obsidian** — read and append to notes, if either is part of the user's actual
  workflow (both are commonly cited by Hermes's own integration list).
- [ ] **Home Assistant** — control smart-home devices, if applicable to the user's setup. Purely
  optional; only worth it if there is real hardware to talk to.
- [ ] **Image generation** — `image_generate`-style tool calling an external model API. Judged
  lower priority than the above because it is a novelty capability rather than a daily-use one for
  a personal assistant; revisit if there's a concrete use case.
- [ ] **Text-to-speech** — generate a voice reply. Only useful once (or if) Kumo also handles voice
  *input* — see kumo's ROADMAP.md, which defers voice notes until a transcription MCP tool exists.
  TTS output without STT input is an odd half of the feature.

## Explicitly not on this list

- **Web search, news, crypto prices, PDF/Excel/paper reading, database access** — already built.
- **Command execution, file read/write on the Kumo host, scheduling, memory** — these are Kumo's
  own native responsibilities (see kumo's ROADMAP.md), not something this server should also expose;
  a second implementation of the same capability would just create two places to keep in sync.
- **Speech-to-text** — not on this list *yet* only because there's no concrete need driving it right
  now; add it here once voice note support in Kumo is actually being built.
