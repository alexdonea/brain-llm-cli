"""src/live_brain.py - watch the mind think, IN THE TERMINAL. No server, no browser, no dependency.

`brain-llm <agent> live` draws an ASCII brain whose regions light up (ANSI truecolor) in the REAL call order
of a command's pathway - appraisal → salience → neuromods → emotion → global-workspace ignition → encode -
beside a live dashboard of EVERY state variable: the PAD mood, the current emotion, all seven neuromodulators,
the HPA stress cascade, the global workspace, and the memory counts. Pure stdlib: ANSI escapes + time.sleep +
in-place redraw. Animates only on a real TTY; piped/`--frame` prints one static frame (so it stays testable).

The pathway→region map is the truth of how src/runtime.py runs each loop; the VALUES shown are the agent's
real, current state. Functional model of a mind, drawn as one - not a claim of felt experience.
"""
from __future__ import annotations
import json as _json
import os
import sys
import time

import brain as B

# ── palette (category → RGB), matching the web preview the colours were designed in ──────────────────
COL = {
    "coral":  (216, 90, 48),   "amber": (186, 117, 23), "pink": (212, 83, 126),
    "purple": (127, 119, 221), "green": (99, 153, 34),  "teal": (29, 158, 117), "blue": (55, 138, 221),
}
IDLE = (91, 102, 117)          # dim slate for a region that isn't firing
RULE = (70, 79, 96)            # outline / separators
TEXT = (157, 167, 179)
HEAD = (123, 136, 150)

# ── the ~10 regions, placed on a 46×17 brain canvas (centre col, row, category colour) ───────────────
REGIONS = {
    "ws": ("workspace", 23, 2,  "purple"),
    "ap": ("appraise",  10, 5,  "coral"),
    "gr": ("graph",     37, 5,  "teal"),
    "em": ("emotion",   16, 8,  "pink"),
    "re": ("retrieve",  35, 8,  "teal"),
    "sa": ("salience",  10, 11, "coral"),
    "hi": ("hippo",     36, 11, "teal"),
    "mo": ("mood",      18, 13, "pink"),
    "se": ("self",      27, 13, "green"),
    "nm": ("neuromod",  23, 15, "amber"),
}
WB, HB = 46, 17                 # brain canvas size

# ── the pathway each command really runs (region id, detail template over the live snapshot) ─────────
FLOWS = {
    "react": [
        ("ap", "§1 appraisal() → PAD v{v:+.2f} a{a:.2f} d{d:.2f}"),
        ("nm", "§2 neuromods() → dopamine {da:.2f}"),
        ("sa", "§3 salience() → {last_sal:.2f}"),
        ("em", "§9 emotion() → \"{emo}\""),
        ("ws", "§12 workspace() → {ign}"),
        ("se", "§23 self_relevance()"),
        ("mo", "§7 update_mood() → v{v:+.2f}"),
        ("hi", "§8 encode() → {nepi} episodes"),
    ],
    "recall": [
        ("re", "§4 base_activation() ACT-R d=0.5"),
        ("gr", "§27 graph_proximity() spreading"),
        ("mo", "§7 mood-congruent bias"),
        ("ws", "§12 workspace() → top-K surfaced"),
    ],
    "sleep": [
        ("hi", "§20 sleep() → replay episodes"),
        ("gr", "§8 consolidate() → facts promoted"),
        ("nm", "§18 neuromods reset → calm"),
        ("em", "§33 REM depotentiation"),
        ("ws", "§5 forget() → low-salience pruned"),
    ],
}


# ── event log: a real mind-event appends one line here; a running `live` watcher tails it to light up ─────
# Working memory (scratch, gitignored) - never the persistent brain. Mind-event commands → the flow they run.
FLOW_FOR = {"react": "react", "remember": "react", "recall": "recall", "know": "recall", "sleep": "sleep"}


def _act_path(root):
    return os.path.join(str(root), "working", "activations.jsonl")


def record(brain, cmd, label=""):
    """If `cmd` is a real mind-event, append one activation line so a live watcher can animate it. Best-effort
    and capped (working memory is scratch); NEVER raises - the CLI must not break if optional viz logging hiccups."""
    flow = FLOW_FOR.get(cmd)
    if not flow:
        return
    try:
        p = _act_path(brain.root)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        old = open(p).read().splitlines()[-99:] if os.path.exists(p) else []   # cap the log
        old.append(_json.dumps({"flow": flow, "cmd": cmd, "label": str(label)[:60], "t": time.time()}))
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(old) + "\n")
        os.replace(tmp, p)
    except Exception:
        pass


def last_activation(root):
    """The most recent activation record (or None). `live` compares its `t` to detect NEW brain activity."""
    try:
        recs = [l for l in (x.strip() for x in open(_act_path(root))) if l]
        return _json.loads(recs[-1]) if recs else None
    except Exception:
        return None


def _color_on():
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _fg(rgb, s, bold=False):
    if not _color_on():
        return s
    b = "1;" if bold else ""
    return f"\x1b[{b}38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{s}\x1b[0m"


def snapshot(brain):
    """Every state variable the dashboard shows, pulled from the agent's real, current brain."""
    nm, hpa, ws = brain.neuromods, brain.hpa, brain.workspace
    last = brain.episodes[-1] if brain.episodes else {}
    return {
        "name": brain.name or "(unnamed)",
        "v": brain.mood.valence, "a": brain.mood.arousal, "d": brain.mood.dominance,
        "mood": B.label_affect(brain.mood)["word"], "emo": B.label_affect(brain.emotion)["word"],
        "da": nm.da, "ne": nm.ne, "ach": nm.ach, "sero": nm.serotonin, "oxy": nm.oxytocin,
        "cort": nm.cortisol, "ne_tonic": nm.ne_tonic,
        "hpa_cort": hpa.cortisol, "crh": hpa.crh, "acth": hpa.acth,
        "ignited": bool(ws.get("ignited")), "ign": "IGNITED ✦" if ws.get("ignited") else "quiet",
        "focus": (ws.get("focus") or "-"),
        "nepi": len(brain.episodes), "nfacts": len(brain.facts),
        "nskills": len(brain.efficacy), "ngoals": len(brain.self_model.goals),
        "last_feel": (last.get("feeling", {}) or {}).get("word", "-"),
        "last_sal": float(last.get("salience", 0.0) or 0.0),
    }


def _bar(val, lo, hi, color, width=12):
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (val - lo) / (hi - lo)))
    n = int(round(frac * width))
    return _fg(color, "█" * n) + _fg(RULE, "░" * (width - n))


def _brain_base():
    """Plain outline rows (each padded to WB visible cols); regions are overlaid, coloured, at render."""
    g = [[" "] * WB for _ in range(HB)]
    import math
    cx, cy, rx, ry = 22.5, 8.0, 21.5, 7.6
    for t in range(0, 720):
        ang = math.radians(t / 2.0)
        x, y = int(round(cx + rx * math.cos(ang))), int(round(cy + ry * math.sin(ang)))
        if 0 <= x < WB and 0 <= y < HB and g[y][x] == " ":
            g[y][x] = "·"
    for y in range(3, HB - 2):                                  # faint longitudinal fissure
        xx = int(round(cx + 2.0 * math.sin(y)))
        if 0 <= xx < WB and g[y][xx] == " ":
            g[y][xx] = ":"
    rows = ["".join(r) for r in g]
    for _, (label, cc, ry_, _c) in REGIONS.items():            # blank the label slots so overlay is clean
        s = cc - len(label) // 2
        rows[ry_] = rows[ry_][:s] + " " * len(label) + rows[ry_][s + len(label):]
    return rows


_BASE = None


def _brain_rows(lit):
    global _BASE
    if _BASE is None:
        _BASE = _brain_base()
    out = []
    by_row = {}
    for rid, (label, cc, ry_, col) in REGIONS.items():
        by_row.setdefault(ry_, []).append((cc - len(label) // 2, label, col, rid))
    for y in range(HB):
        if y not in by_row:
            out.append(_fg(RULE, _BASE[y]))
            continue
        plain = _BASE[y]
        segs, cur = [], 0
        for start, label, col, rid in sorted(by_row[y]):
            segs.append(_fg(RULE, plain[cur:start]))
            if rid == lit:
                segs.append(_fg(COL[col], label, bold=True))
            else:
                segs.append(_fg(IDLE, label))
            cur = start + len(label)
        segs.append(_fg(RULE, plain[cur:]))
        out.append("".join(segs))
    return out


def _dash(snap, flow=""):
    """The right-hand dashboard: every variable, as labelled bars + values."""
    def row(lbl, val, lo, hi, color, fmt="{:.2f}"):
        return f" {_fg(HEAD, lbl.ljust(9))}{_bar(val, lo, hi, color)} {_fg(TEXT, fmt.format(val))}"

    def head(t):
        return " " + _fg(HEAD, "─ " + t + " " + "─" * max(0, 21 - len(t)))

    ign = _fg(COL['purple'], "IGNITED ✦", bold=True) if snap['ignited'] else _fg(IDLE, "quiet")
    asleep = flow == "sleep" or snap.get('ach', 1.0) < 0.5     # acetylcholine ~1.0 wake/encode, ~0.1 consolidate (§2)
    state = _fg(COL['purple'], "⏾ consolidating", bold=True) if asleep else _fg(COL['green'], "● awake")
    rows = [
        " " + _fg(COL['purple'], snap['name'], bold=True) + _fg(HEAD, "  ·  ") + state,
        head("mood (PAD)"),
        row("valence", snap['v'], -1, 1, COL['pink'], "{:+.2f}"),
        row("arousal", snap['a'], 0, 1, COL['pink']),
        row("dominanc", snap['d'], 0, 1, COL['pink']),
        f" {_fg(HEAD, 'feeling'.ljust(9))}{_fg(TEXT, snap['mood'])}  →  emotion {_fg(TEXT, snap['emo'])}",
        head("neuromodulators"),
        row("dopamine", snap['da'], 0, 1, COL['amber']),
        row("noradren", snap['ne'], 0, 1, COL['amber']),
        row("acetylch", snap['ach'], 0, 1, COL['amber']),
        row("serotonn", snap['sero'], 0, 1, COL['amber']),
        row("oxytocin", snap['oxy'], 0, 1, COL['amber']),
        row("cortisol", snap['cort'], 0, 1, COL['amber']),
        head("stress (HPA)"),
        row("cortisol", snap['hpa_cort'], 0, 1, COL['coral']),
        row("crh→acth", snap['acth'], 0, 1, COL['coral']),
        head("workspace"),
        f" {_fg(HEAD, 'ignited'.ljust(9))}{ign}",
        f" {_fg(HEAD, 'focus'.ljust(9))}{_fg(TEXT, str(snap['focus'])[:24])}",
        head("memory"),
        f" {_fg(HEAD, 'episodes'.ljust(9))}{_fg(TEXT, str(snap['nepi']))}   {_fg(HEAD, 'facts ')}{_fg(TEXT, str(snap['nfacts']))}",
        f" {_fg(HEAD, 'skills'.ljust(9))}{_fg(TEXT, str(snap['nskills']))}   {_fg(HEAD, 'goals ')}{_fg(TEXT, str(snap['ngoals']))}",
        f" {_fg(HEAD, 'last'.ljust(9))}{_fg(TEXT, snap['last_feel'])}  sal {_fg(TEXT, format(snap['last_sal'], '.2f'))}",
    ]
    return rows


def _frame(snap, lit, flow, trace):
    brain = _brain_rows(lit)
    dash = _dash(snap, flow)
    n = max(len(brain), len(dash))
    lines = [_fg(HEAD, f" brain-llm · live  ") + _fg(RULE, "─" * 26) + _fg(COL['teal'], f"[ {flow} ]")]
    for i in range(n):
        left = brain[i] if i < len(brain) else ""
        # pad left to WB *visible* cols (ANSI is zero-width, base rows are exactly WB plain chars)
        padplain = WB - (len(B_strip(left)))
        right = dash[i] if i < len(dash) else ""
        lines.append("  " + left + " " * max(0, padplain) + "   " + right)
    lines.append("")
    lines.append("  " + _fg(HEAD, "firing ") + (trace[0] if trace else ""))
    lines.append("  " + _fg(HEAD, "trace  ") + _fg(IDLE, "  ·  ".join(trace[1:4])))
    return lines


def B_strip(s):
    """Visible length helper: strip ANSI SGR sequences."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\x1b":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def _paint(lines, first):
    sys.stdout.write(("\x1b[H" if not first else ""))
    sys.stdout.write("\n".join(l + "\x1b[K" for l in lines) + "\x1b[J")
    sys.stdout.flush()


def _enter_cbreak():
    """Put the terminal in cbreak mode so single keys read instantly (no Enter). ISIG stays on, so Ctrl-C still
    interrupts. Returns saved settings to restore, or None where termios is unavailable (non-POSIX)."""
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return (fd, old)
    except Exception:
        return None


def _restore_term(saved):
    if saved:
        try:
            import termios
            termios.tcsetattr(saved[0], termios.TCSADRAIN, saved[1])
        except Exception:
            pass


def _wait_key(timeout):
    """Block up to `timeout`s for ONE keystroke; return it, or None on timeout (POSIX select on stdin)."""
    try:
        import select
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
    except Exception:
        time.sleep(timeout)
    return None


SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _footer(mode, flow="", spin="", trigger=""):
    def k(key, lbl):
        return _fg(COL["teal"], key, bold=True) + _fg(HEAD, " " + lbl + "   ")
    if mode == "idle":
        status = _fg(IDLE, "○ idle  ") + _fg(HEAD, "waiting for activity  ") + _fg(COL["teal"], spin)
    elif mode == "demo":
        status = _fg(COL["amber"], "▶ demo  ") + _fg(HEAD, "looping " + flow)
    else:
        status = _fg(COL["green"], "● live  ") + _fg(HEAD, flow + "  ") + (_fg(IDLE, "◂ " + trigger[:30]) if trigger else "")
    return "  " + status + _fg(RULE, "    ·    ") + k("q", "quit") + k("r/c/s", "demo a flow")


def animate(brain, flow="react", once=False, frame=False, demo=False, step=0.55):
    """Draw the live brain in the terminal.

    DEFAULT (a TTY) = WATCH mode: the brain sits IDLE (a dim breathing dashboard) and lights up ONLY when a
    REAL mind-event happens - it tails the agent's activation log (written by `react`/`recall`/`know`/`sleep`,
    even from another terminal) and animates that exact pathway with the post-event state. `q`/Ctrl-C quits;
    `r`/`c`/`s` play a flow on demand. `--demo` loops a pathway continuously (no waiting); `--once` plays it
    once; piped / `--frame` prints one static frame (so it stays testable)."""
    flow = flow if flow in FLOWS else "react"
    tty = sys.stdout.isatty() and sys.stdin.isatty() and not frame
    root = getattr(brain, "root", None)
    QUIT, SWITCH = ("q", "Q", "\x03", "\x1b"), {"r": "react", "c": "recall", "s": "sleep"}

    def trace_for(seq, snap, upto):
        return [_fg(COL[REGIONS[rid][3]], det.format(**snap), bold=True) for rid, det in reversed(seq[:upto])]

    def reload():
        return type(brain)(root=root) if root else brain

    def play(fl, mode, trigger, first):
        """Animate one full pass of flow `fl` over freshly-read state. Returns ('quit' | 'switch:x' | 'done', first)."""
        b = reload(); snap = snapshot(b); seq = FLOWS.get(fl, FLOWS["react"])
        for i, (rid, _d) in enumerate(seq):
            _paint(_frame(snap, rid, fl, trace_for(seq, snap, i + 1)) + ["", _footer(mode, fl, trigger=trigger)], first); first = False
            k = _wait_key(step)
            if k in QUIT:
                return "quit", first
            if k in SWITCH:
                return "switch:" + SWITCH[k], first
        _paint(_frame(snap, None, fl, trace_for(seq, snap, len(seq))) + ["", _footer(mode, fl, trigger=trigger)], first)
        return "done", first

    if not tty:                                                # piped / --frame: one static frame
        snap = snapshot(brain)
        print("\n".join(_frame(snap, "ws", flow, trace_for(FLOWS[flow], snap, len(FLOWS[flow]))) + ["", _footer("demo", flow)]))
        return

    saved = _enter_cbreak()
    sys.stdout.write("\x1b[2J\x1b[?25l")                        # clear + hide cursor
    first = True
    try:
        if demo or once:                                       # ── DEMO / ONCE: replay, no waiting on activity ──
            cur = flow
            while True:
                res, first = play(cur, "demo" if demo else "live", "", first)
                if res == "quit" or once:
                    return
                if res.startswith("switch:"):
                    cur = res.split(":")[1]; continue
                k = _wait_key(1.2)
                if k in QUIT:
                    return
                if k in SWITCH:
                    cur = SWITCH[k]
        # ── WATCH (default): idle until a REAL mind-event fires, then animate THAT pathway ──
        last = last_activation(root)
        last_t = last.get("t", 0.0) if last else 0.0
        spin_i = 0
        while True:
            cur = last_activation(root)
            if cur and cur.get("t", 0.0) > last_t:             # NEW activity → light up its flow
                last_t = cur["t"]
                res, first = play(cur.get("flow", "react"), "live", cur.get("label", ""), first)
                if res == "quit":
                    return
                if res.startswith("switch:"):                  # user grabbed control mid-event
                    res, first = play(SWITCH.get(res.split(":")[1], "react"), "live", "", first)
                    if res == "quit":
                        return
                _wait_key(0.5)
            else:                                              # idle → dim brain, breathing dashboard, spinner
                snap = snapshot(reload())
                _paint(_frame(snap, None, "idle", []) + ["", _footer("idle", spin=SPIN[spin_i % len(SPIN)])], first); first = False
                spin_i += 1
                k = _wait_key(0.4)
                if k in QUIT:
                    return
                if k in SWITCH:                                # play a flow on demand while idle
                    res, first = play(SWITCH[k], "live", "", first)
                    if res == "quit":
                        return
    except KeyboardInterrupt:
        pass
    finally:
        _restore_term(saved)
        sys.stdout.write("\x1b[?25h\n")                        # restore cursor
        sys.stdout.flush()


if __name__ == "__main__":                                    # tiny manual check against an agent dir
    sys.path.insert(0, os.path.dirname(__file__))
    from runtime import Brain
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "agents", "default", "memory")
    animate(Brain(root=root), flow=sys.argv[2] if len(sys.argv) > 2 else "react",
            once="--once" in sys.argv, demo="--demo" in sys.argv, frame="--frame" in sys.argv)
