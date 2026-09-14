"""Personal config template for eagle_status.py.

Copy this file to config_local.py (same directory) and fill in your own
values. config_local.py is gitignored -- it's never pushed to the shared
repo, so each person's PCSS identity stays private and local.
"""

# Your PCSS SSH login, e.g. "yara-sh@eagle.man.poznan.pl"
SSH_HOST = "your-username@eagle.man.poznan.pl"

# Path to the SSH private key that logs into the host above.
from pathlib import Path
SSH_KEY = Path.home() / ".ssh" / "id_ed25519_psnc"

# Your PCSS account(s) (from `sacctmgr show assoc user=$USER`), used to
# decide which running jobs count as "mine" in the usage/cost table.
MY_ACCOUNTS = ("your-account-01",)

# Must match the username in SSH_HOST -- used for sacct usage/cost lookups.
MY_USERNAME = "your-username"

# Optional: a teammate's PCSS username to highlight separately (their GPUs
# get their own border/fill color, distinct from yours and from everyone
# else). Leave as your own username (or comment out and set to MY_USERNAME)
# if you don't want to track anyone else.
TRACKED_USER = "your-username"
