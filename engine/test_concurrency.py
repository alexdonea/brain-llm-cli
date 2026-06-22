"""
test_concurrency.py — the per-agent advisory lock actually serializes concurrent load→modify→save.

The durability fixes made each store write atomic (no torn files), but two runs could still interleave:
both load N episodes, each appends one, each saves N+1 → one update silently lost. The lock in agent.py
(_acquire_agent_lock) closes that window. This test proves it by firing many concurrent CLI `react`s — each
a separate process doing a full load→append→save — at the SAME agent, then asserting EVERY one survived.

It exercises real OS processes + real files (no mocks), so it is the one slow test in the suite (~a few seconds).
"""
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "agent.py")

PROCS = 4        # concurrent writers
PER_PROC = 10    # sequential reacts each → expect PROCS * PER_PROC episodes, with NONE lost


def _cli(home, *args):
    """Run one agent.py CLI invocation against an isolated BRAIN_HOME, returning the CompletedProcess."""
    env = dict(os.environ, BRAIN_HOME=home)
    return subprocess.run([sys.executable, AGENT, *args], env=env,
                          capture_output=True, text=True, timeout=60)


def _hammer(home, agent_name, n, tag):
    """Fire n sequential CLI reacts at one agent; each is its own process = its own load→modify→save."""
    for i in range(n):
        # react takes positional: event valence goal_relevance control
        r = _cli(home, "react", f"{tag}-{i}", "0.1", "0.5", "0.5", "--agent", agent_name)
        assert r.returncode == 0, f"react failed: {r.stderr or r.stdout}"


def test_concurrent_reacts_lose_no_episodes():
    home = tempfile.mkdtemp(prefix="brain-conc-")
    try:
        # Bootstrap the agent once (avoid racing on first-run creation), THEN unleash the writers.
        boot = _cli(home, "create", "racer")
        assert boot.returncode == 0, f"create failed: {boot.stderr or boot.stdout}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=PROCS) as pool:
            futs = [pool.submit(_hammer, home, "racer", PER_PROC, f"w{p}") for p in range(PROCS)]
            for f in futs:
                f.result()

        events = os.path.join(home, "agents", "racer", "memory", "episodic", "events.jsonl")
        with open(events) as fh:
            count = sum(1 for line in fh if line.strip())

        # With the lock every append is serialized → all PROCS*PER_PROC episodes persist. Without it, concurrent
        # saves clobber each other and count would come out short (lost updates).
        assert count == PROCS * PER_PROC, (
            f"expected {PROCS * PER_PROC} episodes, got {count} — concurrent saves lost updates")
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    test_concurrent_reacts_lose_no_episodes()
    print(f"PASS: {PROCS * PER_PROC} concurrent reacts, zero lost episodes")
