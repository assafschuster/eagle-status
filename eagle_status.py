#!/usr/bin/env python3
"""Visual snapshot of Eagle Proxima H100 cluster status.

SSHes to Eagle, queries sinfo + reservations + my squeue, then renders a PNG
showing each GPU node as a cell (darkness = GPU utilization 0/4 -> 4/4) with
border color encoding state (drain, reserved, idle).

Run modes:
    python3 eagle_status.py             # one-shot, opens PNG once
    python3 eagle_status.py --watch 30  # live terminal dashboard, refresh
                                        # every 30s. PNG also refreshed on
                                        # disk as a side effect.
"""
import argparse
import re
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

OUT_PNG = Path.home() / "Desktop" / "eagle_proxima_status.png"
OUT_HTML = Path.home() / "Desktop" / "eagle_proxima_status.html"
SSH_HOST = "assafschuster@eagle.man.poznan.pl"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519_psnc"
MY_ACCOUNTS = ("pl0827-01", "pl0910-01")
MY_USERNAME = "assafschuster"   # matches SSH_HOST's login -- used for sacct usage/cost lookups
# Teammate(s) to highlight separately from "mine" and from everyone else --
# e.g. so it's obvious at a glance which GPUs a specific collaborator is
# using, the same way "my" GPUs already get their own border color.
TRACKED_USER = "yara-sh"

# Both rates are now confirmed real PSNC prices (as of 2026-09-07) -- note
# they're in DIFFERENT currencies and are deliberately NOT converted/summed
# into one blended figure, since that would require guessing an exchange
# rate (introducing exactly the kind of unconfirmed number this is trying
# to avoid). GPU cost and CPU cost are reported side by side, each in its
# own real currency.
GPU_RATE_EUR = 2.0    # per GPU-hour
CPU_RATE_PLN = 1.0    # per CPU-hour (Polish zloty)


def ssh(cmd: str) -> str:
    """Run cmd via ssh and return stdout. Raises on error."""
    full = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-i", str(SSH_KEY), SSH_HOST, cmd,
    ]
    r = subprocess.run(full, capture_output=True, text=True, check=True, timeout=30)
    return r.stdout


def parse_nodes(text: str) -> list[dict]:
    """Parse `scontrol -o show node` output into per-node dicts.
    Each node is one line with `Key=Value` pairs separated by spaces."""
    nodes = []
    for line in text.strip().splitlines():
        if not line.startswith("NodeName="):
            continue
        kv = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
        nodes.append(kv)
    return nodes


def parse_reservations(text: str) -> list[dict]:
    """Parse `scontrol -o show reservations` output (one per line)."""
    rsvs = []
    for line in text.strip().splitlines():
        if not line.startswith("ReservationName="):
            continue
        kv = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
        rsvs.append(kv)
    return rsvs


def expand_nodelist(spec: str) -> set[str]:
    """Expand a Slurm host range like 'gpu[47-62,65,67-70]' into individual names."""
    out = set()
    # split top-level commas not inside brackets
    parts = []
    depth = 0
    cur = ""
    for ch in spec:
        if ch == "[":
            depth += 1
            cur += ch
        elif ch == "]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)

    for part in parts:
        m = re.match(r"([a-zA-Z]+)\[([0-9,\-]+)\]", part)
        if not m:
            out.add(part)
            continue
        prefix, ranges = m.group(1), m.group(2)
        for r in ranges.split(","):
            if "-" in r:
                lo, hi = r.split("-")
                width = len(lo)
                for i in range(int(lo), int(hi) + 1):
                    out.add(f"{prefix}{str(i).zfill(width)}")
            else:
                out.add(f"{prefix}{r}")
    return out


# Unified semantic-state palette — applies to BOTH H100 and proxima-cpu grids.
# Same state = same color. Grids distinguished by header + position.
PALETTE = {
    "free":     "#2ca02c",   # green — schedulable
    "llm":      "#8e44ad",   # purple — llm-reserved (H100 only)
    "reserved": "#ff7f00",   # bright orange — any other reservation
    "maint":    "#7e57c2",   # dark violet — maintenance (distinct from llm-purple, far from drain-red)
    "drain":    "#d62728",   # red — admin DRAIN / FAIL only
    "down":     "#555555",   # dark gray — DOWN+NOT_RESPONDING (cluster failure, unreachable)
    "myjob":    "#0000d7",   # deep blue — matches H100's MY_BG (ANSI 27)
    "tracked":  "#e91e8c",   # magenta/pink — TRACKED_USER's GPUs (distinct from all of the above)
}


def gpu_alloc(kv: dict) -> tuple[int, int]:
    """Return (allocated_gpus, total_gpus) for a node, or (0, 0) if no GPU."""
    total = 0
    alloc = 0
    cfg = kv.get("CfgTRES", "")
    for m in re.finditer(r"gres/gpu(?::h100)?=(\d+)", cfg):
        total = int(m.group(1))
        break
    alloc_tres = kv.get("AllocTRES", "")
    for m in re.finditer(r"gres/gpu(?::h100)?=(\d+)", alloc_tres):
        alloc = int(m.group(1))
        break
    return alloc, total


def cpu_alloc(kv: dict) -> tuple[int, int]:
    """Return (allocated_cpus, total_cpus) for a node."""
    total = int(kv.get("CPUTot", 0) or 0)
    alloc = int(kv.get("CPUAlloc", 0) or 0)
    return alloc, total


def categorize(kv: dict) -> str:
    """Bucket H100 node state for border coloring (original H100 behavior —
    do NOT add MAINT here, it would recolor reserved-with-maint nodes blue).
    For CPU-specific categorization that distinguishes MAINT, use
    categorize_cpu()."""
    state = kv.get("State", "")
    if "DRAIN" in state or "DOWN" in state:
        return "drain"
    if "RESERVED" in state:
        return "reserved"
    return "free"


def categorize_cpu(kv: dict) -> str:
    """CPU-grid categorization: distinguishes MAINT from RESERVED/DRAIN/DOWN
    so the proxima-cpu dashboard tells you WHY a node is unavailable.
    DRAIN and FAIL = admin-marked drain (red). DOWN-without-DRAIN =
    unresponsive / cluster failure (gray). On proxima-cpu today, 34 of 60
    nodes are DOWN+NOT_RESPONDING — without this split they'd masquerade
    as 'drain' and overstate admin action vs. cluster sickness."""
    state = kv.get("State", "")
    if "DRAIN" in state or "FAIL" in state:
        return "drain"
    if "DOWN" in state:
        return "down"
    if "MAINT" in state:
        return "maint"
    if "RESERVED" in state:
        return "reserved"
    return "free"


def draw(nodes, rsvs, myjobs, blocking_by_rsv, llm_nodes,
         my_gpu_nodes=None, my_cpu_per_node=None, my_job_nodes=None,
         my_gpu_counts=None, tracked_gpu_counts=None, tracked_cpu_per_node=None,
         my_usage_cost=None, tracked_usage_cost=None):
    if my_gpu_nodes is None: my_gpu_nodes = set()
    if my_cpu_per_node is None: my_cpu_per_node = {}
    if my_job_nodes is None: my_job_nodes = set()
    if my_gpu_counts is None: my_gpu_counts = {}
    if tracked_gpu_counts is None: tracked_gpu_counts = {}
    if tracked_cpu_per_node is None: tracked_cpu_per_node = {}
    if my_usage_cost is None: my_usage_cost = {}
    if tracked_usage_cost is None: tracked_usage_cost = {}
    fig = plt.figure(figsize=(15, 13), dpi=120)
    # 4 rows: H100 grid+summary, proxima-cpu grid+summary, usage/cost, jobs.
    # Reservations table dropped — the grid already encodes reservation
    # status via border color (orange=reserved, purple=llm, red=drain).
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 2],
                          height_ratios=[8, 6, 2, 3],
                          hspace=0.55, wspace=0.18)
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_summary = fig.add_subplot(gs[0, 1])
    ax_cpu_grid = fig.add_subplot(gs[1, 0])
    ax_cpu_summary = fig.add_subplot(gs[1, 1])
    ax_usage = fig.add_subplot(gs[2, :])
    ax_jobs = fig.add_subplot(gs[3, :])
    ax_rsv = None   # reservations no longer rendered; kept var for compat

    # --- Node grid ---
    gpu_nodes = [n for n in nodes if "h100" in n.get("Gres", "")]
    gpu_nodes.sort(key=lambda n: int(re.search(r"\d+", n["NodeName"]).group()))

    cols = 13
    rows = (len(gpu_nodes) + cols - 1) // cols

    for idx, n in enumerate(gpu_nodes):
        r, c = idx // cols, idx % cols
        y = rows - 1 - r
        x = c
        alloc, total = gpu_alloc(n)
        cat = categorize(n)
        # llm reservation override — distinct purple to highlight the
        # 23-node "lost continent" locked to pl0428-01 until 2027.
        if n["NodeName"] in llm_nodes:
            cat = "llm"
        # mine = GPUs on this node that are MINE (0 if none / gpu:0 job)
        mine = my_gpu_counts.get(n["NodeName"], 0) if my_gpu_counts else 0
        # tracked = GPUs on this node held by TRACKED_USER (0 if none/not applicable)
        tracked = tracked_gpu_counts.get(n["NodeName"], 0) if tracked_gpu_counts else 0
        # Fill = proportional dark-green bar, bottom-up (alloc/total of the
        # cell height is green; the rest is left white). Replaces the old
        # continuous grayscale fill, which read as a washed-out "light green"
        # at partial utilization and didn't clearly separate from the green
        # "schedulable" border.
        fill_frac = alloc / total if total else 0
        GPU_FILL_GREEN = "#1b5e20"  # dark green -- distinct from PALETTE["free"] border green
        # Blue BORDER = my NODE (#blue-bordered == my node count, incl gpu:0);
        # the blue COUNT box below = my GPUs (#blue == my GPU count). Magenta
        # BORDER = TRACKED_USER holds GPUs here (only checked when it isn't
        # already my node -- mine takes priority if a node is somehow both).
        is_my_node = n["NodeName"] in my_job_nodes
        is_tracked_node = (not is_my_node) and tracked > 0
        if is_my_node:
            edge = PALETTE["myjob"]
        elif is_tracked_node:
            edge = PALETTE["tracked"]
        else:
            edge = PALETTE.get(cat, "#888888")
        cell_x, cell_y, cell_w, cell_h = x + 0.08, y + 0.08, 0.84, 0.84
        fill_top = cell_y + cell_h * fill_frac  # abs y where green stops (white above)
        if fill_frac > 0:
            ax_grid.add_patch(Rectangle((cell_x, cell_y), cell_w, cell_h * fill_frac,
                                        facecolor=GPU_FILL_GREEN, edgecolor="none", zorder=1))
        rect = Rectangle((cell_x, cell_y), cell_w, cell_h,
                         facecolor="none",
                         edgecolor=edge,
                         linewidth=3.0 if (is_my_node or is_tracked_node) else 2.5,
                         zorder=2)
        ax_grid.add_patch(rect)
        # Node label -- color picked per-label based on whether IT specifically
        # sits over the green fill or the white remainder (the two labels can
        # now land on different backgrounds when a node is partially used).
        nid = n["NodeName"].replace("gpu", "")
        node_label_y = y + 0.66
        label_color = "white" if node_label_y < fill_top else "black"
        ax_grid.text(x + 0.5, node_label_y, nid, ha="center", va="center",
                     fontsize=8, color=label_color, fontweight="bold")
        # Count line: "mine/total" in blue when I have GPUs here; "tracked/total"
        # in magenta when TRACKED_USER does (and I don't); else "alloc/total".
        if mine > 0:
            ax_grid.text(x + 0.5, y + 0.30, f"{mine}/{total}", ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold",
                         bbox=dict(boxstyle="square,pad=0.12", facecolor=PALETTE["myjob"],
                                   edgecolor="none"))
        elif tracked > 0:
            ax_grid.text(x + 0.5, y + 0.30, f"{tracked}/{total}", ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold",
                         bbox=dict(boxstyle="square,pad=0.12", facecolor=PALETTE["tracked"],
                                   edgecolor="none"))
        else:
            count_label_y = y + 0.32
            count_color = "white" if count_label_y < fill_top else "black"
            ax_grid.text(x + 0.5, count_label_y, f"{alloc}/{total}", ha="center",
                         va="center", fontsize=7, color=count_color)
        # Reservation-blocking marker (small diamond in corner)
        if n["NodeName"] in blocking_by_rsv:
            ax_grid.plot(x + 0.85, y + 0.85, marker="D", color=PALETTE["reserved"],
                         markersize=6, markeredgecolor="black",
                         markeredgewidth=0.5)

    ax_grid.set_xlim(0, cols)
    ax_grid.set_ylim(0, rows)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks([]); ax_grid.set_yticks([])
    ax_grid.set_title(f"Proxima H100 nodes — {len(gpu_nodes)} total  "
                      "(green fill height = fraction of GPUs in use; ◆ = blocked-by-rsv; "
                      "blue = my GPUs)",
                      fontsize=9)
    # Solid-color legend swatches — match CPU legend style for visual consistency.
    h100_legend = [
        patches.Patch(facecolor=PALETTE["free"], edgecolor="black", linewidth=1, label="schedulable"),
        patches.Patch(facecolor=PALETTE["llm"], edgecolor="black", linewidth=1, label="llm-reserved (pl0428-01)"),
        patches.Patch(facecolor=PALETTE["reserved"], edgecolor="black", linewidth=1, label="other-reserved"),
        patches.Patch(facecolor=PALETTE["drain"], edgecolor="black", linewidth=1, label="drain/down"),
        patches.Patch(facecolor="white", edgecolor=PALETTE["myjob"], linewidth=2.5, label="my GPUs (blue border + count)"),
        patches.Patch(facecolor="white", edgecolor=PALETTE["tracked"], linewidth=2.5, label=f"{TRACKED_USER}'s GPUs"),
    ]
    ax_grid.legend(handles=h100_legend, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8,
                   frameon=False, handlelength=2.5, handleheight=1.6)

    # --- Summary stats ---
    total_h100 = sum(gpu_alloc(n)[1] for n in gpu_nodes)
    used_h100 = sum(gpu_alloc(n)[0] for n in gpu_nodes)
    drain_nodes = [n for n in gpu_nodes if categorize(n) == "drain"]
    reserved_nodes = [n for n in gpu_nodes if categorize(n) == "reserved"]
    blocked_h100 = sum(gpu_alloc(n)[1] for n in gpu_nodes
                       if n["NodeName"] in blocking_by_rsv
                       or categorize(n) == "drain")
    schedulable_nodes = [n for n in gpu_nodes if categorize(n) == "free"
                         and n["NodeName"] not in blocking_by_rsv]
    free_h100_sched = sum(gpu_alloc(n)[1] - gpu_alloc(n)[0]
                          for n in schedulable_nodes)
    fully_idle = sum(1 for n in schedulable_nodes if gpu_alloc(n)[0] == 0)
    fully_idle_h100 = fully_idle * 4

    ax_summary.axis("off")
    lines = [
        f"H100 TOTALS                     ",
        f"  Cluster H100s          {total_h100:>4}",
        f"  In use right now       {used_h100:>4}  ({100*used_h100/total_h100:.0f}%)",
        f"  Drain/down nodes       {len(drain_nodes):>4} = {len(drain_nodes)*4} H100s",
        f"  llm reservation        {len(llm_nodes):>4} = {len(llm_nodes)*4} H100s (pl0428-01)",
        f"  Other reservations     {max(0, len(blocking_by_rsv)-len(llm_nodes)):>4} H100 nodes",
        f"  Schedulable free       {free_h100_sched:>4}  (across {len(schedulable_nodes)} nodes)",
        f"  Fully-idle nodes (4/4) {fully_idle:>4} = {fully_idle_h100} H100s",
        f"",
        f"FRAGMENTATION INDEX",
        f"  Cluster utilization     {100*used_h100/total_h100:.0f}% but ",
        f"  Fully-idle nodes only   {fully_idle}/{len(gpu_nodes)} ",
        f"  -> 2x4 jobs need 2 fully-idle nodes",
        f"  -> 4x2 jobs need 4 nodes w/ >=2 free",
    ]
    ax_summary.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=10,
                    transform=ax_summary.transAxes)

    # --- Proxima-CPU grid (same conventions as H100, yellow borders) ---
    cpu_nodes = [n for n in nodes if "proxima-cpu" in n.get("Partitions", "")]
    cpu_nodes.sort(key=lambda n: int(re.search(r"\d+", n["NodeName"]).group()))

    cpu_cols = 13
    cpu_rows = (len(cpu_nodes) + cpu_cols - 1) // cpu_cols
    CPU_EDGE = {
        "free":     PALETTE["free"],
        "reserved": PALETTE["reserved"],
        "maint":    PALETTE["maint"],
        "drain":    PALETTE["drain"],
        "down":     PALETTE["down"],
    }
    MY_BLUE = PALETTE["myjob"]   # marker color for "my job is here"

    cpu_busy_nodes = []  # alloc > 0
    cpu_idle_sched = []  # alloc == 0, category free
    cpu_blocked = []     # drain/maint/reserved
    for idx, n in enumerate(cpu_nodes):
        r, c = idx // cpu_cols, idx % cpu_cols
        y = cpu_rows - 1 - r
        x = c
        alloc, total = cpu_alloc(n)
        cat = categorize_cpu(n)
        if cat not in CPU_EDGE:
            cat = "free"
        if cat == "free":
            (cpu_busy_nodes if alloc > 0 else cpu_idle_sched).append(n)
        else:
            cpu_blocked.append(n)
        fill_frac = alloc / total if total else 0
        # Same darkness rule as H100: 0 alloc → white, full → black.
        gray = 1.0 - fill_frac
        # "my job" overrides border color to blue; TRACKED_USER's job (when
        # the node isn't already mine) overrides it to magenta instead.
        is_my_cpu_node = n["NodeName"] in my_cpu_per_node
        is_tracked_cpu_node = (not is_my_cpu_node) and n["NodeName"] in tracked_cpu_per_node
        if is_my_cpu_node:
            edge = PALETTE["myjob"]
        elif is_tracked_cpu_node:
            edge = PALETTE["tracked"]
        else:
            edge = CPU_EDGE[cat]
        rect = Rectangle((x + 0.08, y + 0.08), 0.84, 0.84,
                         facecolor=(gray, gray, gray),
                         edgecolor=edge,
                         linewidth=3.0 if (is_my_cpu_node or is_tracked_cpu_node) else 2.5)
        ax_cpu_grid.add_patch(rect)
        nid = re.sub(r"^[a-zA-Z]+", "", n["NodeName"])  # e.g. "e2412" → "2412"
        label_color = "white" if fill_frac > 0.55 else "black"
        ax_cpu_grid.text(x + 0.5, y + 0.66, nid, ha="center", va="center",
                         fontsize=7, color=label_color, fontweight="bold")
        ax_cpu_grid.text(x + 0.5, y + 0.32, f"{alloc}/{total}",
                         ha="center", va="center",
                         fontsize=6, color=label_color)

    ax_cpu_grid.set_xlim(0, cpu_cols)
    ax_cpu_grid.set_ylim(0, cpu_rows)
    ax_cpu_grid.set_aspect("equal")
    ax_cpu_grid.set_xticks([]); ax_cpu_grid.set_yticks([])
    ax_cpu_grid.set_title(
        f"Proxima-CPU nodes — {len(cpu_nodes)} total  "
        "(fill darkness = CPU cores in use; border color = state)",
        fontsize=9)
    cpu_legend = [
        patches.Patch(facecolor=PALETTE["free"],     edgecolor="black", linewidth=1, label="schedulable"),
        patches.Patch(facecolor=PALETTE["reserved"], edgecolor="black", linewidth=1, label="reserved"),
        patches.Patch(facecolor=PALETTE["maint"],    edgecolor="black", linewidth=1, label="maintenance"),
        patches.Patch(facecolor=PALETTE["drain"],    edgecolor="black", linewidth=1, label="drain/fail"),
        patches.Patch(facecolor=PALETTE["down"],     edgecolor="black", linewidth=1, label="down/unresponsive"),
        patches.Patch(facecolor=PALETTE["myjob"],    edgecolor="black", linewidth=1, label="my job here"),
        patches.Patch(facecolor=PALETTE["tracked"],  edgecolor="black", linewidth=1, label=f"{TRACKED_USER}'s job"),
    ]
    ax_cpu_grid.legend(handles=cpu_legend, loc="upper center",
                       bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9,
                       frameon=False, handlelength=2.5, handleheight=1.6)

    # --- Proxima-CPU summary (same layout style as H100 totals) ---
    total_cpu_cores = sum(cpu_alloc(n)[1] for n in cpu_nodes)
    used_cpu_cores = sum(cpu_alloc(n)[0] for n in cpu_nodes)
    n_drain = sum(1 for n in cpu_nodes if categorize_cpu(n) == "drain")
    n_down  = sum(1 for n in cpu_nodes if categorize_cpu(n) == "down")
    n_maint = sum(1 for n in cpu_nodes if categorize_cpu(n) == "maint")
    n_resv  = sum(1 for n in cpu_nodes if categorize_cpu(n) == "reserved")
    n_sched = len(cpu_busy_nodes) + len(cpu_idle_sched)
    n_idle  = len(cpu_idle_sched)
    pct_used = (100 * used_cpu_cores / total_cpu_cores) if total_cpu_cores else 0
    ax_cpu_summary.axis("off")
    cpu_lines = [
        f"PROXIMA-CPU TOTALS",
        f"  Cluster nodes          {len(cpu_nodes):>4}",
        f"  Cores in use right now {used_cpu_cores:>4} / {total_cpu_cores}  ({pct_used:.0f}%)",
        f"  Drain/fail nodes       {n_drain:>4}",
        f"  Down/unresponsive      {n_down:>4}",
        f"  Maintenance nodes      {n_maint:>4}",
        f"  Reserved nodes         {n_resv:>4}",
        f"  Schedulable nodes      {n_sched:>4}",
        f"  Fully-idle (0 cores)   {n_idle:>4}",
    ]
    ax_cpu_summary.text(0.0, 1.0, "\n".join(cpu_lines), va="top", ha="left",
                        family="monospace", fontsize=10,
                        transform=ax_cpu_summary.transAxes)

    # --- Reservations: NOT rendered (grid encodes via border color) ---

    # --- GPU/CPU usage & cost (sacct, trailing day/week/month) ---
    ax_usage.axis("off")

    def _usage_row(label, cost):
        d = cost.get("day", {}); w = cost.get("week", {}); m = cost.get("month", {})
        return [label,
                f"{d.get('gpu_hr', 0):.2f}", f"{d.get('cpu_hr', 0):.1f}",
                f"€{d.get('gpu_cost_eur', 0):.2f}", f"{d.get('cpu_cost_pln', 0):.2f}zł",
                f"{w.get('gpu_hr', 0):.2f}", f"{w.get('cpu_hr', 0):.1f}",
                f"€{w.get('gpu_cost_eur', 0):.2f}", f"{w.get('cpu_cost_pln', 0):.2f}zł",
                f"{m.get('gpu_hr', 0):.2f}", f"{m.get('cpu_hr', 0):.1f}",
                f"€{m.get('gpu_cost_eur', 0):.2f}", f"{m.get('cpu_cost_pln', 0):.2f}zł"]

    usage_rows = [
        ["User", "Day\nGPU-hr", "Day\nCPU-hr", "Day\n€ (GPU)", "Day\nzł (CPU)",
         "Week\nGPU-hr", "Week\nCPU-hr", "Week\n€ (GPU)", "Week\nzł (CPU)",
         "Month\nGPU-hr", "Month\nCPU-hr", "Month\n€ (GPU)", "Month\nzł (CPU)"],
        _usage_row(f"Me ({MY_USERNAME})", my_usage_cost),
        _usage_row(TRACKED_USER, tracked_usage_cost),
    ]
    tbl_usage = ax_usage.table(cellText=usage_rows, loc="upper left", cellLoc="center",
                               colWidths=[0.10, 0.075, 0.075, 0.08, 0.08, 0.075, 0.075, 0.08, 0.08, 0.075, 0.075, 0.08, 0.08])
    tbl_usage.auto_set_font_size(False)
    tbl_usage.set_fontsize(8)
    tbl_usage.scale(1, 1.8)
    for i in range(len(usage_rows[0])):
        tbl_usage[0, i].set_facecolor("#dddddd")
        tbl_usage[0, i].set_text_props(fontweight="bold")
    ax_usage.set_title(
        "GPU/CPU usage & cost (sacct, trailing windows from now — both rates are confirmed real "
        "PSNC prices: €2.00/GPU-hour, 1 zł/CPU-hour — shown separately, not converted/summed, "
        "since combining currencies would need a guessed exchange rate)",
        fontsize=9, loc="left")

    # --- My jobs ---
    ax_jobs.axis("off")
    job_rows = [["Job", "Name", "State", "Nodes", "GPUs", "Part", "Time", "Reason / Nodelist"]]
    for j in myjobs:
        # Nodes and GPUs are SEPARATE columns — a gpu:0 job reads Nodes=5 GPUs=0.
        job_rows.append([j.get("jobid", "?"), j.get("name", "?")[:18],
                         j.get("state", "?"), str(j.get("nodes", "?")),
                         str(j.get("gpus", 0)), j.get("kind", "?"),
                         j.get("time", "?"), j.get("reason", "?")[:38]])
    if len(job_rows) == 1:
        job_rows.append(["(no queued or running jobs)", "", "", "", "", "", "", ""])
    tbl = ax_jobs.table(cellText=job_rows, loc="upper left", cellLoc="left",
                      colWidths=[0.10, 0.16, 0.09, 0.06, 0.05, 0.05, 0.08, 0.41])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for i in range(len(job_rows[0])):
        tbl[0, i].set_facecolor("#dddddd")
        tbl[0, i].set_text_props(fontweight="bold")
    ax_jobs.set_title("My jobs", fontsize=10, loc="left")

    fig.suptitle(f"Eagle Proxima cluster status — {dt.datetime.now():%Y-%m-%d %H:%M:%S %Z}",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_squeue(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        if line.startswith("JOBID"):
            continue
        parts = line.split("|", 6)
        if len(parts) < 7:
            continue
        # SLURM wraps reasons in parens like "(Resources)" — strip them.
        reason = parts[6].strip()
        if reason.startswith("(") and reason.endswith(")"):
            reason = reason[1:-1]
        # Tag partition as GPU vs CPU so the dashboard row tells the user
        # what kind of node count they're looking at.
        partition = parts[3]
        if "proxima-cpu" in partition:
            kind = "CPU"
        elif "proxima" in partition:
            kind = "GPU"
        else:
            kind = partition[:3].upper()
        rows.append({"jobid": parts[0], "state": parts[1], "name": parts[2],
                     "partition": partition, "kind": kind,
                     "nodes": parts[4], "time": parts[5], "reason": reason})
    return rows


# ANSI colors for terminal rendering. 256-color codes.
ANSI_RESET = "\x1b[0m"
# Full clear including scrollback (3J), then home, then erase (2J).
ANSI_CLEAR = "\x1b[3J\x1b[H\x1b[2J"
ANSI_HIDE_CURSOR = "\x1b[?25l"
ANSI_SHOW_CURSOR = "\x1b[?25h"
# Per-category palette: bg fills the "used GPU" cells; fg colors the box
# border (so the category is visible even when the node is idle/white).
CAT_BG = {
    "free":     "\x1b[48;5;28m",    # green
    "llm":      "\x1b[48;5;91m",    # purple
    "reserved": "\x1b[48;5;208m",   # bright orange
    "maint":    "\x1b[48;5;99m",    # dark violet (distinct from llm-purple, not red-adjacent)
    "drain":    "\x1b[48;5;196m",   # bright red — admin DRAIN/FAIL only
    "down":     "\x1b[48;5;240m",   # dark gray — DOWN+NOT_RESPONDING (cluster failure)
}
# Explicit bright-white background — for "GPU is free / available" cells
# anywhere in the dashboard. Same color in node interiors AND in the
# partition bar, so the eye learns "white = available".
WHITE_BG = "\x1b[48;5;231m"
# Bold bright-white fg for the digit shown on a filled rectangle.
DIGIT_STYLE = "\x1b[1m\x1b[97m"
CAT_FG = {
    "free":     "\x1b[38;5;82m",    # bright green border
    "llm":      "\x1b[38;5;141m",   # bright purple border
    "reserved": "\x1b[38;5;208m",   # bright orange border
    "maint":    "\x1b[38;5;105m",   # lighter violet border (pairs with bg 99)
    "drain":    "\x1b[38;5;196m",   # bright red border
    "down":     "\x1b[38;5;245m",   # light gray border (pairs with bg 240)
}
# Color used for the GPU slots inside a cell that are running MY jobs.
# Only affects the interior fill; the cell border still shows category color.
MY_BG = "\x1b[48;5;27m"     # deep blue
CAT_LABEL = {
    "free":     "free",
    "llm":      "llm",
    "reserved": "rsv",
    "maint":    "maint",
    "drain":    "drain",
    "down":     "down",
}

# Proxima-CPU uses the SAME palette as H100 (CAT_FG / CAT_BG) — same semantic
# state must show the same color in both grids. The grids are distinguished
# by their headers + position in the dashboard, not by colors.
CPU_FG = CAT_FG
CPU_BG = CAT_BG


MY_FG = "\x1b[1m\x1b[38;5;27m"   # deep-blue bold — border color for "my job here"
TRACKED_BG = "\x1b[48;5;200m"    # magenta — TRACKED_USER's GPU slots inside a cell
TRACKED_FG = "\x1b[1m\x1b[38;5;200m"   # magenta bold — border color for "TRACKED_USER's node"


def render_cell(n: dict, llm_nodes: set, my_gpus_per_node: dict,
                my_job_nodes: set = None, tracked_gpus_per_node: dict = None) -> tuple[str, str, str, str]:
    """Render one node as 4 ANSI-escaped strings (top, mid1, mid2, bottom).
    Heavy bold border in category color. Interior is 4 chars wide, 2 lines
    tall, edge-to-edge: N chars colored (= N GPUs used) + (4-N) chars white.
    No padding between border and fill.

        ┏g13━┓
        ┃▓▓  ┃   (2/4 GPUs in use)
        ┃▓▓  ┃
        ┗━━━━┛
    """
    nid = n["NodeName"].replace("gpu", "")
    alloc, total = gpu_alloc(n)
    cat = categorize(n)
    if n["NodeName"] in llm_nodes:
        cat = "llm"
    fg = CAT_FG[cat]
    bg = CAT_BG[cat]
    BOLD = "\x1b[1m"

    INTERIOR_W = 4
    # Interior: 4 chars wide × 2 lines.
    # Used GPUs split into "mine" (blue) + "others" (category color).
    # Free GPUs in white (or x-marked if node is blocked).
    n_filled = alloc if total else 0
    n_free = INTERIOR_W - n_filled
    n_mine = min(n_filled, my_gpus_per_node.get(n["NodeName"], 0))
    n_tracked = min(n_filled - n_mine, (tracked_gpus_per_node or {}).get(n["NodeName"], 0))
    n_others = max(0, n_filled - n_mine - n_tracked)
    # TWO consistencies:
    #  (1) blue BORDER = my NODE  -> #blue-bordered cells == my node count
    #      (row "5 nodes" <=> 5 blue borders, incl. gpu:0 jobs).
    #  (2) blue BLOCKS = my GPUs  -> #blue blocks == my GPU count, summed across
    #      blue cells == my total GPUs (row "0 gpu" <=> 0 blue blocks).
    # Same pair of consistencies for magenta = TRACKED_USER, checked only when
    # the node isn't already mine (mine takes priority on the border).
    is_my_node = bool(my_job_nodes and n["NodeName"] in my_job_nodes)
    is_tracked_node = (not is_my_node) and n_tracked > 0
    if is_my_node:
        fg = MY_FG
    elif is_tracked_node:
        fg = TRACKED_FG

    label = f"g{nid}"
    if len(label) > INTERIOR_W:
        label = label[:INTERIOR_W]
    label_fill = INTERIOR_W - len(label)
    top_interior = label + ("━" * label_fill)
    top = f"{BOLD}{fg}┏{top_interior}┓{ANSI_RESET}"

    # ---- info line (digits + at most one x) ----
    info_mine = ""
    if n_mine > 0:
        info_mine = (MY_BG + DIGIT_STYLE + str(n_mine) +
                     (" " * (n_mine - 1)) + ANSI_RESET)
    info_tracked = ""
    if n_tracked > 0:
        info_tracked = (TRACKED_BG + DIGIT_STYLE + str(n_tracked) +
                        (" " * (n_tracked - 1)) + ANSI_RESET)
    info_others = ""
    if n_others > 0:
        info_others = (bg + DIGIT_STYLE + str(n_others) +
                       (" " * (n_others - 1)) + ANSI_RESET)
    info_free = ""
    if n_free > 0:
        if cat == "free":
            info_free = WHITE_BG + (" " * n_free) + ANSI_RESET
        else:
            info_free = (WHITE_BG + "\x1b[1m\x1b[38;5;16m" +
                         ("x" * n_free) + ANSI_RESET)

    # ---- plain line (same fills, no markers) ----
    plain_mine = (MY_BG + (" " * n_mine) + ANSI_RESET) if n_mine > 0 else ""
    plain_tracked = (TRACKED_BG + (" " * n_tracked) + ANSI_RESET) if n_tracked > 0 else ""
    plain_others = (bg + (" " * n_others) + ANSI_RESET) if n_others > 0 else ""
    plain_free = (WHITE_BG + (" " * n_free) + ANSI_RESET) if n_free > 0 else ""

    mid1 = f"{BOLD}{fg}┃{ANSI_RESET}{info_mine}{info_tracked}{info_others}{info_free}{BOLD}{fg}┃{ANSI_RESET}"
    mid2 = f"{BOLD}{fg}┃{ANSI_RESET}{plain_mine}{plain_tracked}{plain_others}{plain_free}{BOLD}{fg}┃{ANSI_RESET}"

    bottom = f"{BOLD}{fg}┗{'━' * INTERIOR_W}┛{ANSI_RESET}"
    return top, mid1, mid2, bottom


def render_cpu_cell(n: dict, my_cpu_per_node: dict, tracked_cpu_per_node: dict = None) -> tuple[str, str, str, str]:
    """Render one proxima-cpu node as 4 ANSI strings (top, mid1, mid2, bottom).
    Same 4-line bordered-cell convention as render_cell, but for CPU cores
    instead of GPUs:
      - 4-char-wide interior; fill = ceil(alloc/total * 4) cells in CPU_BG
      - remainder cells = white (free)
      - my_cpu_per_node: dict {node_name: cpus_allocated_to_me} — used to
        proportionally fill the cell (my-cores blue, others' cores in
        category color, free cores white) and to override border color.
      - tracked_cpu_per_node: same idea for TRACKED_USER (magenta), checked
        only when the node isn't already mine.
      - border color encoded by category (yellow=schedulable, blue=maint,
        orange=resv, red=drain)
    """
    if tracked_cpu_per_node is None:
        tracked_cpu_per_node = {}
    nid = re.sub(r"^[a-zA-Z]+", "", n["NodeName"])   # "e2412" → "2412"
    alloc, total = cpu_alloc(n)
    cat = categorize_cpu(n)
    if cat not in CPU_FG:
        cat = "free"
    my_cores = my_cpu_per_node.get(n["NodeName"], 0)
    is_mine = my_cores > 0
    tracked_cores = 0 if is_mine else tracked_cpu_per_node.get(n["NodeName"], 0)
    is_tracked = tracked_cores > 0
    # "my job" / "tracked job" override only the BORDER color (so it's
    # spottable). Interior fill is proportional: my-cores blue, tracked-cores
    # magenta, OTHER users' cores keep the category bg color, free cores stay
    # white. This way the cell tells the user "how much of this node is
    # actually mine / TRACKED_USER's".
    if is_mine:
        fg = "\x1b[38;5;27m"
    elif is_tracked:
        fg = TRACKED_FG
    else:
        fg = CPU_FG[cat]
    bg = CPU_BG[cat]
    BOLD = "\x1b[1m"
    INTERIOR_W = 4

    label = nid[-4:]
    label_fill = INTERIOR_W - len(label)
    top_interior = label + ("━" * label_fill)
    top = f"{BOLD}{fg}┏{top_interior}┓{ANSI_RESET}"

    # Interior: 4 chars × 2 lines.
    # mid1 = proportional color fill (mine/tracked/others/free in real proportions).
    # mid2 = numeric "alloc/total" — bold black on white — so the user can
    #        read off the exact core count without counting blocks.
    other_cores = max(0, alloc - my_cores - tracked_cores)
    # Round each band to an integer number of 4 cells; reconcile any drift
    # by handing the remainder to the free band.
    n_mine_fill = 0 if total == 0 else min(INTERIOR_W, round(my_cores * INTERIOR_W / total))
    n_tracked_fill = 0 if total == 0 else min(INTERIOR_W - n_mine_fill,
                                               round(tracked_cores * INTERIOR_W / total))
    n_others    = 0 if total == 0 else min(INTERIOR_W - n_mine_fill - n_tracked_fill,
                                            round(other_cores * INTERIOR_W / total))
    # If any cores are alloc'd but rounding sent every band to 0, force the
    # largest real band to show at least 1 cell.
    if alloc > 0 and n_mine_fill == 0 and n_tracked_fill == 0 and n_others == 0:
        largest = max(my_cores, tracked_cores, other_cores)
        if largest == my_cores:
            n_mine_fill = 1
        elif largest == tracked_cores:
            n_tracked_fill = 1
        else:
            n_others = 1
    n_free = INTERIOR_W - n_mine_fill - n_tracked_fill - n_others

    # Unavailable nodes (drain/down/maint/reserved, not mine/tracked): stamp big black X.
    is_unavail = cat != "free" and not is_mine and not is_tracked
    if is_unavail:
        mid1 = f"{BOLD}{fg}┃{ANSI_RESET}{bg}{BOLD}\x1b[38;5;16mX  X{ANSI_RESET}{BOLD}{fg}┃{ANSI_RESET}"
        mid2 = f"{BOLD}{fg}┃{ANSI_RESET}{bg}{BOLD}\x1b[38;5;16m XX {ANSI_RESET}{BOLD}{fg}┃{ANSI_RESET}"
    else:
        # mid1: proportional fill — mine in blue, tracked in magenta, others in cat-bg, free in white.
        fill_my      = (MY_BG      + " " * n_mine_fill    + ANSI_RESET) if n_mine_fill    else ""
        fill_tracked = (TRACKED_BG + " " * n_tracked_fill + ANSI_RESET) if n_tracked_fill else ""
        fill_other   = (bg         + " " * n_others       + ANSI_RESET) if n_others       else ""
        fill_free    = (WHITE_BG   + " " * n_free         + ANSI_RESET) if n_free         else ""
        mid1 = f"{BOLD}{fg}┃{ANSI_RESET}{fill_my}{fill_tracked}{fill_other}{fill_free}{BOLD}{fg}┃{ANSI_RESET}"
        # mid2: alloc count, bold black on white — fits 0–999 in 4 chars.
        count_str = f"{alloc:>4d}"
        mid2 = (f"{BOLD}{fg}┃{ANSI_RESET}"
                f"{WHITE_BG}\x1b[1m\x1b[38;5;16m{count_str}{ANSI_RESET}"
                f"{BOLD}{fg}┃{ANSI_RESET}")

    bottom = f"{BOLD}{fg}┗{'━' * INTERIOR_W}┛{ANSI_RESET}"
    return top, mid1, mid2, bottom


def render_tui(nodes_data, rsvs_data, myjobs, blocking, llm_nodes,
               my_gpus_per_node, gpu_util_rows, refresh_secs,
               my_cpu_per_node=None, my_job_nodes=None, tracked_gpus_per_node=None,
               tracked_cpu_per_node=None, my_usage_cost=None, tracked_usage_cost=None):
    if my_cpu_per_node is None:
        my_cpu_per_node = {}
    if my_job_nodes is None:
        my_job_nodes = set()
    if tracked_gpus_per_node is None:
        tracked_gpus_per_node = {}
    if tracked_cpu_per_node is None:
        tracked_cpu_per_node = {}
    if my_usage_cost is None:
        my_usage_cost = {}
    if tracked_usage_cost is None:
        tracked_usage_cost = {}
    """Render the cluster grid + single-line stacked totals + reservations
    + my jobs. Each node is a 4-line bordered box with edge-to-edge fill."""
    gpu_nodes = [n for n in nodes_data if "h100" in n.get("Gres", "")]
    gpu_nodes.sort(key=lambda n: int(re.search(r"\d+", n["NodeName"]).group()))

    out = [ANSI_CLEAR]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.append(f"\x1b[1mEagle Proxima H100 status — {now}\x1b[0m   "
               f"(refreshes every {refresh_secs}s — Ctrl-C to stop)")
    # No legend here — moved between grid and bar (single unified legend).
    out.append("")

    # ──── Grid: 10 cols × 4 lines per cell ────
    cols = 13
    rows = (len(gpu_nodes) + cols - 1) // cols
    CELL_W = 6  # 1 corner + 4 interior + 1 corner
    GAP = 1
    for r in range(rows):
        l1 = l2 = l3 = l4 = ""
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(gpu_nodes):
                blank = " " * (CELL_W + GAP)
                l1 += blank; l2 += blank; l3 += blank; l4 += blank
                continue
            t, m1, m2, b = render_cell(gpu_nodes[idx], llm_nodes, my_gpus_per_node, my_job_nodes, tracked_gpus_per_node)
            sep = " " * GAP
            l1 += t + sep
            l2 += m1 + sep
            l3 += m2 + sep
            l4 += b + sep
        out.append(l1)
        out.append(l2)
        out.append(l3)
        out.append(l4)
    out.append("")

    # Mutually exclusive buckets, with each blocked category split into
    # busy vs idle so we can see how loaded the llm / drain / rsv reservations
    # actually are. Sum across all buckets = total_h100.
    total_h100 = sum(gpu_alloc(n)[1] for n in gpu_nodes)
    sched_busy = sched_free = 0
    llm_busy = llm_idle = 0
    drain_busy = drain_idle = 0
    rsv_busy = rsv_idle = 0
    for n in gpu_nodes:
        alloc, total = gpu_alloc(n)
        cat = categorize(n)
        if n["NodeName"] in llm_nodes:
            llm_busy += alloc
            llm_idle += (total - alloc)
        elif cat == "drain":
            drain_busy += alloc
            drain_idle += (total - alloc)
        elif cat == "reserved":
            rsv_busy += alloc
            rsv_idle += (total - alloc)
        else:
            sched_busy += alloc
            sched_free += (total - alloc)

    # Bar segments use the same color convention as node interiors:
    #   WHITE = free for me (or blocked-idle if marked with x — same visual rule)
    #   GREEN = busy on schedulable node (will free up later)
    #   PURPLE / RED / ORANGE solid = busy on blocked node
    #   WHITE+x = idle but blocked (same as cell convention)
    # Bar width = grid row width (13 cells × CELL_W + 12 × GAP) so bar visually
    # spans the full grid, with a thin black box border around it.
    BAR_W = cols * CELL_W + (cols - 1) * GAP   # 13*6 + 12*1 = 90
    X_BG = "__X__"  # sentinel: render as WHITE_BG with x's (used only for sched-free)
    # Blocked categories (llm/drain/rsv) collapse busy+idle into ONE segment in
    # their solid category color — the busy/idle split inside a blocked
    # category isn't actionable to the user. Only schedulable splits.
    segments = [
        (sched_free,           WHITE_BG,           f"free {sched_free}"),
        (sched_busy,           CAT_BG["free"],     f"sched-busy {sched_busy}"),
        (llm_busy + llm_idle,  CAT_BG["llm"],      f"llm {llm_busy + llm_idle}"),
        (rsv_busy + rsv_idle,  CAT_BG["reserved"], f"reserved {rsv_busy + rsv_idle}"),
        (drain_busy + drain_idle, CAT_BG["drain"], f"drain {drain_busy + drain_idle}"),
    ]
    bar = ""; used_w = 0
    # Distribute width — last NON-ZERO segment gets the remainder so width sums exactly.
    nonzero_idx = [i for i, s in enumerate(segments) if s[0] > 0]
    last_nz = nonzero_idx[-1] if nonzero_idx else -1
    for i, (count, color, lab) in enumerate(segments):
        if count <= 0:
            continue
        if i == last_nz:
            w = BAR_W - used_w
        else:
            w = max(1, round(count / max(1, total_h100) * BAR_W))
        if color == X_BG:
            bar += WHITE_BG + "\x1b[1m\x1b[38;5;16m" + ("x" * w) + ANSI_RESET
        else:
            bar += color + (" " * w) + ANSI_RESET
        used_w += w
    # Unified legend: bar-segment swatches (with counts) + the two non-segment
    # markers ("my GPU" blue, "blocked-idle" white+x). The "free N" segment
    # already carries the white swatch, so no standalone "free GPU" entry.
    # Always show the FIRST segment (`free N`) even when count is 0 — the
    # user wants to see "free 0" so the leftmost slot is consistent across
    # cluster states. Other zero-count segments are still dropped.
    legend_parts = []
    for i, (count, color, lab) in enumerate(segments):
        if count <= 0 and i > 0:
            continue
        legend_parts.append(f"{color}  {ANSI_RESET} {lab}")
    legend_parts.append(f"{MY_BG}  {ANSI_RESET} my GPU")
    legend_parts.append(f"{MY_FG}┃┃{ANSI_RESET} my-GPU border")
    legend_parts.append(f"{TRACKED_BG}  {ANSI_RESET} {TRACKED_USER} GPU")
    legend_parts.append(f"{TRACKED_FG}┃┃{ANSI_RESET} {TRACKED_USER} border")
    legend_parts.append(f"{WHITE_BG}\x1b[1m\x1b[38;5;16mxx{ANSI_RESET} blocked-idle")
    legend_line = "   ".join(legend_parts)
    # Legend sits ABOVE the bar title (right under the grid).
    out.append(legend_line)
    out.append(f"\x1b[1mCluster H100 partitioning ({total_h100} H100s on {len(gpu_nodes)} nodes):\x1b[0m")
    # Thin black box-drawing border around the bar.
    BLACK_FG = "\x1b[38;5;16m"
    out.append(BLACK_FG + "┌" + ("─" * BAR_W) + "┐" + ANSI_RESET)
    out.append(BLACK_FG + "│" + ANSI_RESET + bar + BLACK_FG + "│" + ANSI_RESET)
    out.append(BLACK_FG + "└" + ("─" * BAR_W) + "┘" + ANSI_RESET)
    out.append("")

    # ──── Proxima-CPU grid (same cell convention as H100, yellow borders) ────
    cpu_nodes = [n for n in nodes_data if "proxima-cpu" in n.get("Partitions", "")]
    cpu_nodes.sort(key=lambda n: int(re.search(r"\d+", n["NodeName"]).group()))
    if cpu_nodes:
        # Title only — single unified legend lives between grid and bar (below).
        out.append(f"\x1b[1mProxima-CPU grid — {len(cpu_nodes)} nodes\x1b[0m")
        cpu_cols_tui = 13
        cpu_rows_tui = (len(cpu_nodes) + cpu_cols_tui - 1) // cpu_cols_tui
        for r in range(cpu_rows_tui):
            l1 = l2 = l3 = l4 = ""
            for c in range(cpu_cols_tui):
                idx = r * cpu_cols_tui + c
                if idx >= len(cpu_nodes):
                    blank = " " * 7   # 6 cell + 1 gap
                    l1 += blank; l2 += blank; l3 += blank; l4 += blank
                    continue
                t, m1, m2, b = render_cpu_cell(cpu_nodes[idx], my_cpu_per_node, tracked_cpu_per_node)
                l1 += t + " "
                l2 += m1 + " "
                l3 += m2 + " "
                l4 += b + " "
            out.append(l1); out.append(l2); out.append(l3); out.append(l4)

        # Proxima-CPU partitioning bar (same visual style as H100 partitioning).
        # Mutually exclusive buckets, each split busy/idle, sum across = total cores.
        total_cpu_cores_tui = sum(cpu_alloc(n)[1] for n in cpu_nodes)
        sched_busy_c = sched_free_c = 0
        rsv_busy_c = rsv_idle_c = 0
        maint_busy_c = maint_idle_c = 0
        drain_busy_c = drain_idle_c = 0
        down_busy_c = down_idle_c = 0
        for n in cpu_nodes:
            a, t = cpu_alloc(n)
            cat = categorize_cpu(n)
            if cat == "drain": drain_busy_c += a; drain_idle_c += (t - a)
            elif cat == "down": down_busy_c += a; down_idle_c += (t - a)
            elif cat == "maint": maint_busy_c += a; maint_idle_c += (t - a)
            elif cat == "reserved": rsv_busy_c += a; rsv_idle_c += (t - a)
            else: sched_busy_c += a; sched_free_c += (t - a)

        # Bar width matches CPU grid row width (13*CELL_W + 12 gaps = 90) so it
        # visually aligns with the row of cells above, exactly like the H100 bar.
        BAR_W_C = cpu_cols_tui * CELL_W + (cpu_cols_tui - 1) * GAP
        # Same scheme as H100: schedulable splits busy/free; blocked categories
        # show as ONE solid segment in their color.
        segs_c = [
            (sched_free_c,                  WHITE_BG,           f"free {sched_free_c}"),
            (sched_busy_c,                  CAT_BG["free"],     f"sched-busy {sched_busy_c}"),
            (rsv_busy_c + rsv_idle_c,       CAT_BG["reserved"], f"reserved {rsv_busy_c + rsv_idle_c}"),
            (maint_busy_c + maint_idle_c,   CAT_BG["maint"],    f"maint {maint_busy_c + maint_idle_c}"),
            (drain_busy_c + drain_idle_c,   CAT_BG["drain"],    f"drain {drain_busy_c + drain_idle_c}"),
            (down_busy_c + down_idle_c,     CAT_BG["down"],     f"down {down_busy_c + down_idle_c}"),
        ]
        # Distribute widths exactly so the bar fills BAR_W_C — last non-zero segment
        # absorbs rounding remainder (matches H100 bar logic).
        bar_c = ""
        used_w_c = 0
        nonzero_idx_c = [i for i, s in enumerate(segs_c) if s[0] > 0]
        last_nz_c = nonzero_idx_c[-1] if nonzero_idx_c else -1
        for i, (cnt, color, _) in enumerate(segs_c):
            if cnt <= 0: continue
            if i == last_nz_c:
                w = BAR_W_C - used_w_c
            else:
                w = max(1, round(cnt / max(1, total_cpu_cores_tui) * BAR_W_C))
            bar_c += color + (" " * w) + ANSI_RESET
            used_w_c += w
        # Unified legend: bar-segment swatches (with counts) + "my job" marker.
        # The "free N" segment already carries the white swatch, so no standalone
        # "free core" entry. The leftmost (free) entry is always shown — even
        # at 0 cores — so the legend's left edge is consistent across states.
        legend_parts_c = []
        for i, (cnt, color, lab) in enumerate(segs_c):
            if cnt <= 0 and i > 0: continue
            legend_parts_c.append(f"{color}  {ANSI_RESET} {lab}")
        legend_parts_c.append(f"{MY_BG}  {ANSI_RESET} my job")
        legend_parts_c.append(f"{TRACKED_BG}  {ANSI_RESET} {TRACKED_USER} job")
        legend_line_c = "   ".join(legend_parts_c)
        # Legend sits ABOVE the bar title (right under the CPU grid).
        out.append(legend_line_c)
        out.append(f"\x1b[1mProxima-CPU partitioning ({total_cpu_cores_tui} cores on {len(cpu_nodes)} nodes):\x1b[0m")
        # Same thin box-drawing border as H100.
        out.append(BLACK_FG + "┌" + ("─" * BAR_W_C) + "┐" + ANSI_RESET)
        out.append(BLACK_FG + "│" + ANSI_RESET + bar_c + BLACK_FG + "│" + ANSI_RESET)
        out.append(BLACK_FG + "└" + ("─" * BAR_W_C) + "┘" + ANSI_RESET)
        out.append("")

    # Reservations table removed — the H100 grid already encodes reservation
    # state via border colors (purple=llm, orange=other-reserved, red=drain).

    # GPU/CPU usage & cost (sacct, trailing day/week/month from now)
    out.append("\x1b[1mGPU/CPU usage & cost\x1b[0m  "
               "(both rates confirmed real PSNC prices: €2.00/GPU-hr, 1 zł/CPU-hr -- "
               "shown separately, not converted/summed):")
    out.append(f"  {'':<16} {'Day GPU-hr':>10} {'CPU-hr':>7} {'€(GPU)':>8} {'zł(CPU)':>9}   "
               f"{'Week GPU-hr':>11} {'CPU-hr':>7} {'€(GPU)':>8} {'zł(CPU)':>9}   "
               f"{'Month GPU-hr':>12} {'CPU-hr':>7} {'€(GPU)':>8} {'zł(CPU)':>9}")
    for label, cost in [(f"Me ({MY_USERNAME})", my_usage_cost), (TRACKED_USER, tracked_usage_cost)]:
        d, w, m = cost.get("day", {}), cost.get("week", {}), cost.get("month", {})
        out.append(
            f"  {label:<16} "
            f"{d.get('gpu_hr', 0):>10.2f} {d.get('cpu_hr', 0):>7.1f} "
            f"{d.get('gpu_cost_eur', 0):>7.2f}€ {d.get('cpu_cost_pln', 0):>7.2f}zł   "
            f"{w.get('gpu_hr', 0):>11.2f} {w.get('cpu_hr', 0):>7.1f} "
            f"{w.get('gpu_cost_eur', 0):>7.2f}€ {w.get('cpu_cost_pln', 0):>7.2f}zł   "
            f"{m.get('gpu_hr', 0):>12.2f} {m.get('cpu_hr', 0):>7.1f} "
            f"{m.get('gpu_cost_eur', 0):>7.2f}€ {m.get('cpu_cost_pln', 0):>7.2f}zł")
    out.append("")

    # My jobs
    out.append("\x1b[1mMy jobs:\x1b[0m")
    if myjobs:
        out.append(f"  {'JOBID':<10} {'STATE':<9} {'NAME':<20} {'NODES':<6} {'GPUS':<5} {'PART':<5} {'TIME':<7} REASON")
        for j in myjobs:
            # Separate, unambiguous columns: NODES and GPUS are different numbers.
            # A gpu:0 staging job reads NODES=5 GPUS=0 PART=GPU — clear, not "5 GPU".
            out.append(f"  {j['jobid']:<10} {j['state']:<9} {j['name'][:20]:<20} "
                       f"{j['nodes']:<6} {j.get('gpus',0):<5} {j.get('kind','?'):<5} "
                       f"{j['time']:<7} {j['reason'][:55]}")
    else:
        out.append("  (no queued or running jobs)")
    out.append("")

    # ---- Live nvidia-smi for MY running GPUs ----
    if gpu_util_rows:
        out.append("\x1b[1mMy GPUs — live nvidia-smi (via srun --overlap):\x1b[0m")
        out.append(f"  {'NODE':<7} {'JOB':<8} {'COMPUTE':>7}  {'HBM-bw':>6}  "
                   f"{'MEM (GB)':>11}  {'POWER (W)':>11}  {'TEMP':>4}  {'SM MHz':>6}")
        for r in gpu_util_rows:
            mem_str = f"{r['mem_used_gb']:.1f}/{r['mem_total_gb']:.0f}"
            pwr_str = f"{r['power_w']:.0f}/{r['power_limit_w']:.0f}"
            out.append(
                f"  {r['node']:<7} {r['job']:<8} "
                f"{r['util_gpu']:>6}%  {r['util_mem']:>5}%  "
                f"{mem_str:>11}  {pwr_str:>11}  "
                f"{r['temp_c']:>3}°C  {r['sm_clk_mhz']:>6}"
            )

    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


def gather_my_gpu_util(my_jobids: list) -> list:
    """Query live nvidia-smi for MY running jobs' GPUs.
    Uses `srun --jobid=N --overlap` which auto-filters to the job's allocation.
    Returns list of dicts: node, job, util_gpu, util_mem, mem_used_gb,
    mem_total_gb, power_w, power_limit_w, temp_c, sm_clk_mhz."""
    QUERY = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
             "power.draw,power.limit,temperature.gpu,clocks.current.sm")
    rows = []
    for jid in my_jobids:
        try:
            cmd = (
                f"timeout 10 srun --jobid={jid} --overlap bash -c "
                f"'H=$(hostname); nvidia-smi --query-gpu={QUERY} "
                f'--format=csv,noheader,nounits | awk -v h="$H" "{{ print h, \\$0 }}"'
                f"' 2>/dev/null"
            )
            text = ssh(cmd)
            for line in text.strip().splitlines():
                # node util_gpu, util_mem, mem_used, mem_total, pwr_draw, pwr_limit, temp, sm_clk
                m = re.match(
                    r"(\S+)\s+(\d+),\s*(\d+),\s*(\d+),\s*(\d+),"
                    r"\s*([\d.]+),\s*([\d.]+),\s*(\d+),\s*(\d+)",
                    line)
                if m:
                    rows.append({
                        "node": m.group(1),
                        "job": jid,
                        "util_gpu": int(m.group(2)),
                        "util_mem": int(m.group(3)),
                        "mem_used_gb": int(m.group(4)) / 1024,
                        "mem_total_gb": int(m.group(5)) / 1024,
                        "power_w": float(m.group(6)),
                        "power_limit_w": float(m.group(7)),
                        "temp_c": int(m.group(8)),
                        "sm_clk_mhz": int(m.group(9)),
                    })
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return rows


def gather_usage_cost(username: str) -> dict:
    """sacct-based GPU-hour / CPU-hour / cost rollup for `username`, over
    three trailing windows anchored at now: day (24h), week (7d), month (30d).
    Each window is CUMULATIVE (e.g. "month" includes everything in "week"
    and "day"), not a distinct non-overlapping bucket -- these are three
    different lookback horizons from right now, not three separate periods.
    Counts elapsed time regardless of job outcome (COMPLETED/CANCELLED/
    TIMEOUT/FAILED all consumed real allocated time)."""
    text = ssh(f'sacct -u {username} -S now-30days -E now --allocations '
               f'--parsable2 --noheader -o Start,AllocTRES,ElapsedRaw')
    now = dt.datetime.now()
    windows = {"day": dt.timedelta(days=1), "week": dt.timedelta(days=7), "month": dt.timedelta(days=30)}
    totals = {w: {"gpu_hr": 0.0, "cpu_hr": 0.0} for w in windows}
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        start_s, alloctres, elapsed_s = parts[0], parts[1], parts[2]
        try:
            start = dt.datetime.strptime(start_s, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        try:
            elapsed = int(elapsed_s)
        except ValueError:
            elapsed = 0
        gpu_m = re.search(r"gres/gpu(?::[a-zA-Z0-9]+)?=(\d+)", alloctres)
        n_gpu = int(gpu_m.group(1)) if gpu_m else 0
        cpu_m = re.search(r"(?:^|,)cpu=(\d+)", alloctres)
        n_cpu = int(cpu_m.group(1)) if cpu_m else 0
        age = now - start
        for wname, wdelta in windows.items():
            if dt.timedelta(0) <= age <= wdelta:
                totals[wname]["gpu_hr"] += n_gpu * elapsed / 3600.0
                totals[wname]["cpu_hr"] += n_cpu * elapsed / 3600.0
    for w in totals:
        g, c = totals[w]["gpu_hr"], totals[w]["cpu_hr"]
        totals[w]["gpu_cost_eur"] = g * GPU_RATE_EUR
        totals[w]["cpu_cost_pln"] = c * CPU_RATE_PLN
    return totals


def gather():
    """Run the SSH queries and parse. Returns
    (nodes, rsvs, myjobs, blocking, llm_nodes, my_gpus_per_node, gpu_util_rows,
     my_cpu_cores_per_node, my_job_nodes, tracked_gpus_per_node, tracked_cpu_per_node,
     my_usage_cost, tracked_usage_cost)."""
    node_text = ssh("scontrol -o show node")
    rsv_text = ssh("scontrol -o show reservations")
    sq_text = ssh('squeue -u $USER --states=PENDING,RUNNING -o "%i|%T|%j|%P|%D|%M|%R" -h')
    # Per-node GPU count for MY currently RUNNING jobs (%b = TRES per node).
    # %C = total CPUs allocated to the job; %b = TRES per node (GPU + cpu).
    # Use both: %C is reliable across SLURM versions for cpu per-node math
    # (we divide by node count); %b is the only way to get :h100:N GPU count.
    my_run_text = ssh('squeue -u $USER -t RUNNING -h -o "%i|%N|%C|%b"')
    tracked_run_text = ssh(f'squeue -u {TRACKED_USER} -t RUNNING -h -o "%N|%C|%b"')

    nodes = parse_nodes(node_text)
    rsvs = parse_reservations(rsv_text)
    myjobs = parse_squeue(sq_text)

    gpu_node_names = {n["NodeName"] for n in nodes if "h100" in n.get("Gres", "")}
    cpu_node_names = {n["NodeName"] for n in nodes if "proxima-cpu" in n.get("Partitions", "")}
    blocking = set()
    for r in rsvs:
        accts = r.get("Accounts", "")
        if not any(a in accts for a in MY_ACCOUNTS):
            in_rsv = expand_nodelist(r.get("Nodes", "")) & gpu_node_names
            blocking |= in_rsv

    llm_nodes = set()
    for r in rsvs:
        if r.get("ReservationName") == "llm":
            llm_nodes = expand_nodelist(r.get("Nodes", "")) & gpu_node_names
            break

    my_gpus_per_node: dict = {}
    my_cpu_cores_per_node: dict = {}   # node -> total CPU cores allocated to ME on that node
    my_job_nodes: set = set()          # EVERY node I have a running job on (any partition, any GPU count)
    my_job_gpus: dict = {}             # jobid -> total GPUs the job holds (0 for CPU / gpu:0 jobs)
    my_running_jobids: list = []
    for line in my_run_text.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        jid, nodelist, total_cpus_s, tres = parts
        if not nodelist.strip():
            continue
        my_running_jobids.append(jid)
        gpu_m = re.search(r"gpu(?::h100)?:(\d+)", tres)
        gpus_per_node = int(gpu_m.group(1)) if gpu_m else 0
        # %C is total CPUs allocated to the whole job; divide by node count
        # to get per-node cpus (assuming even SLURM distribution, the default).
        try:
            total_cpus = int(total_cpus_s)
        except ValueError:
            total_cpus = 0
        nodes_expanded = expand_nodelist(nodelist)
        cpus_per_node = total_cpus // max(1, len(nodes_expanded)) if total_cpus else 0
        my_job_gpus[jid] = gpus_per_node * len(nodes_expanded)
        for nm in nodes_expanded:
            my_job_nodes.add(nm)   # mark the node as "mine" regardless of GPU count
            if gpus_per_node > 0 and nm in gpu_node_names:
                my_gpus_per_node[nm] = my_gpus_per_node.get(nm, 0) + gpus_per_node
            if nm in cpu_node_names and cpus_per_node > 0:
                my_cpu_cores_per_node[nm] = my_cpu_cores_per_node.get(nm, 0) + cpus_per_node

    # Annotate the job rows with their real GPU count so the table can show
    # "5 nodes · 0 GPU" instead of an ambiguous "5 GPU".
    for j in myjobs:
        j["gpus"] = my_job_gpus.get(j.get("jobid", ""), 0)

    gpu_util_rows = gather_my_gpu_util(my_running_jobids) if my_running_jobids else []

    # TRACKED_USER's currently-running GPUs + CPU cores, same parsing pattern
    # as the "mine" block above (job table / per-job GPU accounting isn't
    # needed for them since they only ever get a border/fill highlight, not
    # a jobs table row).
    tracked_gpus_per_node: dict = {}
    tracked_cpu_per_node: dict = {}
    for line in tracked_run_text.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        nodelist, total_cpus_s, tres = parts
        if not nodelist.strip():
            continue
        gpu_m = re.search(r"gpu(?::h100)?:(\d+)", tres)
        gpus_per_node = int(gpu_m.group(1)) if gpu_m else 0
        try:
            total_cpus = int(total_cpus_s)
        except ValueError:
            total_cpus = 0
        nodes_expanded = expand_nodelist(nodelist)
        cpus_per_node = total_cpus // max(1, len(nodes_expanded)) if total_cpus else 0
        for nm in nodes_expanded:
            if gpus_per_node > 0 and nm in gpu_node_names:
                tracked_gpus_per_node[nm] = tracked_gpus_per_node.get(nm, 0) + gpus_per_node
            if nm in cpu_node_names and cpus_per_node > 0:
                tracked_cpu_per_node[nm] = tracked_cpu_per_node.get(nm, 0) + cpus_per_node

    my_usage_cost = gather_usage_cost(MY_USERNAME)
    tracked_usage_cost = gather_usage_cost(TRACKED_USER)

    return (nodes, rsvs, myjobs, blocking, llm_nodes,
            my_gpus_per_node, gpu_util_rows, my_cpu_cores_per_node, my_job_nodes,
            tracked_gpus_per_node, tracked_cpu_per_node,
            my_usage_cost, tracked_usage_cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="Live terminal dashboard, refresh every N seconds. "
                         "If 0 (default), render PNG once and open it.")
    ap.add_argument("--no-png", action="store_true",
                    help="In --watch mode, skip the PNG side-effect.")
    args = ap.parse_args()

    if args.watch <= 0:
        # One-shot: just the PNG.
        (nodes, rsvs, myjobs, blocking, llm_nodes, my_gpus, _util, my_cpu, my_job_nodes,
         tracked_gpus, tracked_cpu, my_cost, tracked_cost) = gather()
        draw(nodes, rsvs, myjobs, blocking, llm_nodes,
             my_gpu_nodes=set(my_gpus.keys()), my_cpu_per_node=my_cpu, my_job_nodes=my_job_nodes, my_gpu_counts=my_gpus,
             tracked_gpu_counts=tracked_gpus, tracked_cpu_per_node=tracked_cpu,
             my_usage_cost=my_cost, tracked_usage_cost=tracked_cost)
        print(f"Wrote {OUT_PNG}", file=sys.stderr)
        subprocess.run(["open", str(OUT_PNG)], check=False)
        return

    # Live terminal dashboard
    sys.stdout.write(ANSI_HIDE_CURSOR)
    try:
        while True:
            try:
                data = gather()  # 13-tuple
                (nodes, rsvs, myjobs, blocking, llm_nodes, my_gpus, util, my_cpu, my_job_nodes,
                 tracked_gpus, tracked_cpu, my_cost, tracked_cost) = data
                render_tui(nodes, rsvs, myjobs, blocking, llm_nodes,
                           my_gpus, util, refresh_secs=args.watch,
                           my_cpu_per_node=my_cpu, my_job_nodes=my_job_nodes,
                           tracked_gpus_per_node=tracked_gpus, tracked_cpu_per_node=tracked_cpu,
                           my_usage_cost=my_cost, tracked_usage_cost=tracked_cost)
                if not args.no_png:
                    draw(nodes, rsvs, myjobs, blocking, llm_nodes,
                         my_gpu_nodes=set(my_gpus.keys()), my_cpu_per_node=my_cpu, my_job_nodes=my_job_nodes, my_gpu_counts=my_gpus,
                         tracked_gpu_counts=tracked_gpus, tracked_cpu_per_node=tracked_cpu,
                         my_usage_cost=my_cost, tracked_usage_cost=tracked_cost)
            except subprocess.CalledProcessError as e:
                sys.stderr.write(f"\n[WARN {dt.datetime.now():%H:%M:%S}] "
                                 f"ssh failed: {e}\n")
            except subprocess.TimeoutExpired:
                sys.stderr.write(f"\n[WARN {dt.datetime.now():%H:%M:%S}] ssh timed out\n")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        sys.stdout.write(ANSI_SHOW_CURSOR)
        print("\nStopped.", file=sys.stderr)
    finally:
        sys.stdout.write(ANSI_SHOW_CURSOR)


if __name__ == "__main__":
    main()
