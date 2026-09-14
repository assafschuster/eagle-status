# eagle-status

Visual monitor for the PCSS Eagle/Proxima H100 cluster. SSHes to Eagle, pulls
`sinfo`/`squeue`/`sacct`, and renders a live grid showing each GPU node's
utilization and state, plus a running usage/cost summary.

## Setup

1. Clone this repo.
2. `pip3 install matplotlib`
3. Copy `config.example.py` to `config_local.py` and fill in your own PCSS
   SSH login, key path, account(s), and username. `config_local.py` is
   gitignored -- it stays local, never pushed.
4. Run it:
   ```
   python3 eagle_status.py             # one-shot PNG, opens once
   python3 eagle_status.py --watch 30  # live terminal dashboard, refreshes every 30s
   ```

## Notes

- `TRACKED_USER` in your `config_local.py` highlights one other PCSS username
  separately (its own border/fill color) so you can see a specific
  collaborator's jobs at a glance alongside your own. Set it to your own
  username if you don't want to track anyone else.
- Requires SSH key-based (passwordless) access to the host in `SSH_HOST`.
