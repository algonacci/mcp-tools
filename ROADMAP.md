# Roadmap

## Near Term

- [ ] Move all diagnostic output from `stdout` to logging or `stderr` to keep
  the MCP stdio JSON-RPC channel clean.
- [ ] Add a configuration status tool that reports which integrations are
  ready, missing configuration, or unavailable without exposing secrets.
- [ ] Add automated tests for partial `.env` configuration, the setup wizard,
  clean startup, and single-interrupt shutdown.

## Architecture

- [ ] Split the large `server.py` into integration modules such as
  `tools/garuda.py`, `tools/email.py`, and `tools/calendar.py`.
- [ ] Keep `mcp-tools` as the complete aggregator while isolating integration
  initialization so one broken integration cannot prevent unrelated tools
  from loading.
- [ ] Rename generic tools such as `search`, `connect`, and `health` to
  integration-specific names.

## Distribution

- [ ] Keep standalone `mcp-*` repositories as focused distributions and define
  a consistent process for syncing their tools into this aggregate server.
- [ ] Document minimal installation and configuration recipes for Windows,
  macOS, and Linux.
