# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Report privately through the repository's **Security** tab ("Report a vulnerability").
Do not open a public issue.

Include the surfbench version, your OS, what goes wrong, and how to reproduce it. Expect an
acknowledgement within a week.

## Threat model

surfbench is a local numerical library and command-line tool.

- **No credentials.** surfbench needs no API keys. The data adapters fetch public data over
  HTTPS from `datasets.tardis.dev` (free first-of-month files) and `www.deribit.com`
  (public market-data endpoint) and nothing else.
- **Downloaded data is untrusted input.** The Tardis extractor streams and decompresses the
  file in memory, parses only rows for the configured instruments, validates column count,
  numeric fields and timestamp order, and writes only the extracted snapshots, atomically.
  Nothing from a download is executed or evaluated. Deribit API responses are parsed as JSON
  and validated field by field.
- **Cache and run directories** hold extracted market data and intermediate results only.
  They are local, gitignored and safe to delete.
