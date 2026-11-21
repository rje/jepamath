"""cloudrun: run jobs on preemptible Vast.ai GPUs with Backblaze B2 as the only durable storage.

Project-agnostic. A project supplies a project config (bucket, credential files, repo) and one TOML
spec per job (command, machine filters, data, budget). See cloud/README.md for the jepamath setup.

Pieces:
  config      project + job specs (TOML), validated on load
  creds       credential files (mode 600) locally, prefixed env vars on an instance
  b2          S3-compatible B2 client, prefix listing, verified directory up/down
  vast        Vast.ai API: offer search and ranking, rent, list, VERIFIED destroy
  sync        instance side: ship log + heartbeat, mirror the run dir to B2, prune rolling checkpoints
  bootstrap.sh instance entrypoint: fetch code at a commit, setup, restore run dir, pull data, run, final flush
  supervisor  local loop: provision, watch heartbeats, replace on preemption/stall/crash, hard spend cap
  cli         python -m cloudrun {offers,plan,launch,status,logs,pull,push-data,instances,destroy}

Money rules (from microlab's paid runs): nothing is rented without --yes; price caps are re-checked
before every rental; spend is capped across all episodes; every destroy is verified against the instance
list; exit status is an explicit sentinel, never inferred from log text.
"""
