#!/usr/bin/env python3
"""
rts_sim.py -- Mini Projects 2.1 / 2.2 / 2.3
Real-Time Systems: RM/EDF simulation, RTA, Priority Inheritance, and
Partitioned Multiprocessor Scheduling.

Run:   python3 rts_sim.py
Out:   console report + figs/*.png
"""
import os, math
from dataclasses import dataclass
from functools import reduce
from math import gcd, ceil, log
from typing import Optional, List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figs"
os.makedirs(FIG, exist_ok=True)
LINE = "=" * 78


# ----------------------------------------------------------------------
# 1. Task model  (Section 2.1)
# ----------------------------------------------------------------------
@dataclass
class Task:
    name: str
    C: int
    T: int
    D: Optional[int] = None

    def __post_init__(self):
        if self.D is None:
            self.D = self.T

    @property
    def U(self) -> float:
        return self.C / self.T

    def __repr__(self):
        return (f"{self.name}(C={self.C},T={self.T},D={self.D},"
                f"u={self.U:.3f})")


def total_utilization(tasks: List[Task]) -> float:
    return sum(t.U for t in tasks)


def rm_bound(n: int) -> float:
    """Liu & Layland bound  n*(2^(1/n) - 1)   (Section 2.2.1)."""
    return n * (2.0 ** (1.0 / n) - 1.0) if n > 0 else 0.0


def hyperperiod(tasks: List[Task]) -> int:
    return reduce(lambda a, b: a * b // gcd(a, b), [int(t.T) for t in tasks], 1)


# ----------------------------------------------------------------------
# 2. Discrete-event uniprocessor simulator (Policies: RM, DM, EDF, LLF)
# ----------------------------------------------------------------------
def simulate(tasks: List[Task], sim_time: int, policy: str = "RM"):
    """Tick-by-tick preemptive simulation. Returns (timeline, completions).
       timeline  : [(t, task_name_or_'idle'), ...]
       completions: [dict(name, release, finish, deadline), ...]"""
    jobs, timeline, completions = [], [], []
    for t in range(sim_time):
        for task in tasks:                                   # release phase
            if t % int(task.T) == 0:
                jobs.append(dict(task=task, release=t,
                                 deadline=t + int(task.D),
                                 remaining=task.C))
        ready = [j for j in jobs if j["remaining"] > 0]
        if not ready:
            timeline.append((t, "idle"))
            continue

        if policy == "RM":                                   # fixed, by period
            key = lambda j: (j["task"].T, j["release"], j["task"].name)
        elif policy == "DM":                                 # fixed, by deadline
            key = lambda j: (j["task"].D, j["release"], j["task"].name)
        elif policy == "EDF":                                # dynamic, by deadline
            key = lambda j: (j["deadline"], j["release"], j["task"].name)
        elif policy == "LLF":                                # dynamic, by laxity
            key = lambda j: (j["deadline"] - t - j["remaining"],
                             j["deadline"], j["task"].name)
        else:
            raise ValueError(policy)

        j = min(ready, key=key)
        j["remaining"] -= 1
        timeline.append((t, j["task"].name))
        if j["remaining"] == 0:
            completions.append(dict(name=j["task"].name, release=j["release"],
                                    finish=t + 1, deadline=j["deadline"]))
    return timeline, completions


def deadline_misses(completions) -> list:
    return [c for c in completions if c["finish"] > c["deadline"]]


def context_switches(timeline) -> int:
    return sum(1 for i in range(1, len(timeline))
               if timeline[i][1] != timeline[i - 1][1])


# ----------------------------------------------------------------------
# 3. Response Time Analysis (Section 2.5.2)
# ----------------------------------------------------------------------
def rta(tasks: List[Task], order: Optional[List[Task]] = None):
    """R_i = C_i + sum_{j in hp(i)} ceil(R_i / T_j) * C_j
       Returns [(task, R, iterations, schedulable), ...] in priority order."""
    if order is None:
        order = sorted(tasks, key=lambda t: (t.T, t.name))    # RM order
    out = []
    for i, ti in enumerate(order):
        hp = order[:i]
        R, it = ti.C, 0
        while True:
            it += 1
            Rn = ti.C + sum(ceil(R / hj.T) * hj.C for hj in hp)
            if Rn == R:
                break
            if Rn > ti.D:
                R = Rn
                break
            R = Rn
            if it > 1000:
                break
        out.append((ti, R, it, R <= ti.D))
    return out


# ----------------------------------------------------------------------
# 4. Gantt helpers
# ----------------------------------------------------------------------
def _runs(timeline):
    segs, t0, cur = {}, 0, timeline[0][1]
    for t in range(1, len(timeline) + 1):
        nxt = timeline[t][1] if t < len(timeline) else None
        if nxt != cur:
            segs.setdefault(cur, []).append((t0, t - t0))
            t0, cur = t, nxt
    return segs


def draw_gantt(ax, timeline, order, title):
    segs = _runs(timeline)
    rows = list(order) + ["idle"]
    ymap = {n: i for i, n in enumerate(rows)}
    pal = plt.cm.tab10.colors
    cmap = {n: pal[i % 10] for i, n in enumerate(order)}
    cmap["idle"] = "#dddddd"
    for name in rows:
        if name in segs:
            ax.broken_barh(segs[name], (ymap[name] - 0.4, 0.8),
                           facecolors=cmap[name], edgecolors="black",
                           linewidth=0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title)
    ax.set_xlabel("time (ticks)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)


# ----------------------------------------------------------------------
# 5. Priority inversion + Priority Inheritance Protocol
# ----------------------------------------------------------------------
def simulate_inversion(scenario, pip: bool, horizon: int = 40):
    """scenario entries: dict(name, release, C, prio, needs_res)
       The whole execution of a 'needs_res' task is its critical section."""
    jobs = [dict(name=s["name"], release=s["release"], C=s["C"],
                 prio=s["prio"], needs_res=s["needs_res"],
                 remaining=s["C"], done=False, finish=None)
            for s in scenario]
    owner, timeline = None, []

    for t in range(horizon):
        # --- who is blocked on the mutex? ---
        blocked = [j for j in jobs
                   if not j["done"] and t >= j["release"]
                   and j["needs_res"] and owner is not None and owner is not j]

        # --- effective priorities (PIP: holder inherits max of blocked) ---
        eff = {j["name"]: j["prio"] for j in jobs}
        if pip and owner is not None and blocked:
            eff[owner["name"]] = max([owner["prio"]] +
                                     [b["prio"] for b in blocked])

        blocked_ids = {id(b) for b in blocked}
        cand = [j for j in jobs
                if not j["done"] and t >= j["release"] and id(j) not in blocked_ids]
        if not cand:
            timeline.append((t, "idle"))
            continue

        j = max(cand, key=lambda x: (eff[x["name"]], x["prio"], -x["release"]))
        if j["needs_res"] and owner is None:          # acquire mutex
            owner = j
        j["remaining"] -= 1
        timeline.append((t, j["name"]))
        if j["remaining"] == 0:
            j["done"] = True
            j["finish"] = t + 1
            if owner is j:
                owner = None                          # release mutex

    return timeline, jobs


# ----------------------------------------------------------------------
# 6. Partitioned multiprocessor bin-packing (Section 2.8.3)
# ----------------------------------------------------------------------
def partition(tasks, m, heuristic="FF"):
    """heuristic in {FF, FFD, BFD, WF}. Admission = per-core Liu&Layland
       bound recomputed for the resulting task count."""
    if heuristic in ("FFD", "BFD"):
        order = sorted(tasks, key=lambda t: -t.U)
    else:
        order = list(tasks)

    cores = [[] for _ in range(m)]
    loads = [0.0] * m

    for task in order:
        cand = []
        for i in range(m):
            n = len(cores[i]) + 1
            newload = loads[i] + task.U
            if newload <= rm_bound(n) + 1e-12:
                cand.append((i, rm_bound(n) - newload))
        if not cand:
            return None, None, cores          # partition failure

        if heuristic == "BFD":
            i = min(cand, key=lambda c: c[1])[0]
        elif heuristic == "WF":
            i = max(cand, key=lambda c: c[1])[0]
        else:
            i = cand[0][0]                    # FF / FFD: first admissible
        cores[i].append(task)
        loads[i] += task.U

    return cores, loads, None


# ======================================================================
#  MAIN
# ======================================================================
def main():
    # ---------------- Section A : worked examples ----------------------
    print(LINE)
    print("MINI PROJECT 2.1 - RM / EDF / PIP SIMULATION")
    print(LINE)

    tset3 = [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12)]
    tset4 = tset3 + [Task("t4", 1, 8)]

    for label, ts in (("3-task example (Sec 2.2.2)", tset3),
                      ("4-task example (Sec 2.3.2)", tset4)):
        U, n, B = total_utilization(ts), len(ts), rm_bound(len(ts))
        print(f"\n{label}:  {[t.name for t in ts]}")
        for t in ts:
            print(f"   {t}  -> u_i = {t.U:.4f}")
        print(f"   Total U          = {U:.4f}")
        print(f"   rm_bound(n={n})   = {B:.4f}")
        print(f"   RM sufficient    : {'PASS' if U <= B else 'FAIL (inconclusive -> use RTA)'}")
        print(f"   EDF exact U<=1   : {'PASS' if U <= 1.0 else 'FAIL'}")

    # ---------------- RM / EDF schedules -------------------------------
    H = hyperperiod(tset4)
    print(f"\nHyperperiod of the 4-task set = {H} ticks")

    for pol in ("RM", "EDF"):
        tl, comp = simulate(tset4, H, pol)
        miss = deadline_misses(comp)
        print(f"\n{pol}: {len(miss)} deadline miss(es)")
        for c in comp:
            flag = "  <-- MISS" if c["finish"] > c["deadline"] else ""
            print(f"   {c['name']}: released {c['release']:>2}, "
                  f"finished {c['finish']:>2}, deadline {c['deadline']:>2}{flag}")

    # ---------------- Response Time Analysis ---------------------------
    print("\n" + LINE)
    print("RESPONSE TIME ANALYSIS (Sec 2.5.2) - 4-task set")
    print(LINE)
    for t, R, it, ok in rta(tset4):
        print(f"   {t.name}: R = {R:>3}, D = {t.D:>3}, iterations = {it}, "
              f"{'SCHEDULABLE' if ok else 'NOT SCHEDULABLE'}")
    print("   => RM CANNOT guarantee tau4 (R4=10 > D4=8) although U=0.875 < 1;")
    print("      EDF is exact and PASSES. This is the RM-vs-EDF contrast.")

    # ---------------- Gantt charts -------------------------------------
    tl_rm, _ = simulate(tset4, H, "RM")
    tl_edf, _ = simulate(tset4, H, "EDF")
    names = [t.name for t in tset4]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    draw_gantt(ax, tl_rm, names, "RM schedule, 4-task set (hyperperiod=24)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_rm.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    draw_gantt(ax, tl_edf, names, "EDF schedule, 4-task set (hyperperiod=24)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_edf.png", dpi=130); plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    draw_gantt(a1, tl_rm, names, "RM")
    draw_gantt(a2, tl_edf, names, "EDF")
    fig.suptitle("Figure 2.1 analogue: RM (top) vs EDF (bottom)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_rm_edf.png", dpi=130); plt.close(fig)

    # ---------------- Lab task: RM bound sweep --------------------------
    ns = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [rm_bound(n) for n in ns], "o-", label="rm_bound(n)")
    ax.axhline(1.0, color="g", ls="--", label="EDF bound = 1.0")
    ax.axhline(log(2), color="r", ls=":", label="ln 2 ~ 0.693")
    ax.set_xlabel("n (tasks)"); ax.set_ylabel("utilization bound")
    ax.set_title("Liu & Layland bound vs n (converges to ln 2)")
    ax.legend(); ax.grid(alpha=.4)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_rm_bound.png", dpi=130); plt.close(fig)

    # ---------------- Lab task: DM vs RM on constrained deadlines -------
    cset = [Task("a", 1, 5, 5), Task("b", 2, 8, 3)]   # D != T for b
    H2 = hyperperiod(cset)
    tl_rm2, _ = simulate(cset, H2, "RM")
    tl_dm2, _ = simulate(cset, H2, "DM")
    print("\nDeadline-Monotonic demo (D_i < T_i for task b):")
    print(f"   RM: {[n for _, n in tl_rm2]}")
    print(f"   DM: {[n for _, n in tl_dm2]}")
    print(f"   schedules differ: {tl_rm2 != tl_dm2}")
    eq = [Task("a", 1, 5), Task("b", 2, 8)]
    print(f"   with all D_i == T_i, RM == DM: "
          f"{simulate(eq, hyperperiod(eq), 'RM')[0] == simulate(eq, hyperperiod(eq), 'DM')[0]}")

    # ---------------- Lab task: LLF thrashing ---------------------------
    print("\nContext switches over one hyperperiod (4-task set):")
    for pol in ("RM", "EDF", "LLF"):
        tl, _ = simulate(tset4, H, pol)
        print(f"   {pol:>4}: {context_switches(tl):>3} switches")

    # ---------------- Priority inversion --------------------------------
    print("\n" + LINE)
    print("PRIORITY INVERSION AND PRIORITY INHERITANCE (Sec 2.6 / 2.7)")
    print(LINE)

    scen = [
        dict(name="L", release=0, C=6, prio=1, needs_res=True),
        dict(name="H", release=1, C=2, prio=3, needs_res=True),
        dict(name="M", release=2, C=4, prio=2, needs_res=False),
    ]
    scen2 = scen + [dict(name="M2", release=3, C=3, prio=2, needs_res=False)]

    results = {}
    for label, sc in (("3-task", scen), ("4-task (+M2)", scen2)):
        for pip in (False, True):
            tl, jobs = simulate_inversion(sc, pip, horizon=40)
            h = next(j for j in jobs if j["name"] == "H")
            blocking = (h["finish"] - h["release"] - h["C"])
            results[(label, pip)] = blocking
            tag = "PIP" if pip else "no protocol"
            print(f"\n{label:>12} | {tag:<12} | H finishes at t={h['finish']:>2} "
                  f"| H blocking = {blocking} ticks")
            print("   " + " ".join(f"{n[0]}{n[1:] if len(n)>1 else ''}"
                                   for _, n in tl[:20]))

    print("\nBound check (Sec 2.7.1): H is blocked by at most ONE critical "
          "section per lower-priority task.")
    print(f"   L critical section = 6 ticks  -> bound = 6")
    print(f"   measured (3-task, PIP)        = {results[('3-task', True)]}  <= 6  OK")
    print(f"   measured (3-task, no proto)   = {results[('3-task', False)]}  > 6  (unbounded inversion)")
    print(f"   measured (4-task, PIP)        = {results[('4-task (+M2)', True)]}  <= 6  OK (bound unaffected)")
    print(f"   measured (4-task, no proto)   = {results[('4-task (+M2)', False)]}  >> 6 (grows with each M)")

    tl_no, _ = simulate_inversion(scen, False)
    tl_pi, _ = simulate_inversion(scen, True)
    iorder = ["L", "M", "H"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    draw_gantt(a1, tl_no, iorder, "Unprotected: H delayed by L's CS AND all of M")
    draw_gantt(a2, tl_pi, iorder, "PIP: L inherits H's priority; H blocked by ONE CS")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_inversion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    draw_gantt(ax, tl_no, iorder, "Unprotected priority inversion")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_inv_nopip.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    draw_gantt(ax, tl_pi, iorder, "Priority Inheritance Protocol")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_inv_pip.png", dpi=130); plt.close(fig)

    # ==================================================================
    # MINI PROJECT 2.2  -- general schedulability comparison tool
    # ==================================================================
    print("\n" + LINE)
    print("MINI PROJECT 2.2 - RM vs EDF SCHEDULABILITY COMPARISON TOOL")
    print(LINE)

    def analyse(ts, name="set"):
        U, n = total_utilization(ts), len(ts)
        bound = rm_bound(n)
        constrained = any(t.D < t.T for t in ts)
        order = sorted(ts, key=lambda t: ((t.D if constrained else t.T), t.name))
        print(f"\n--- {name}  (n={n}, U={U:.4f}, "
              f"{'constrained' if constrained else 'implicit'} deadlines) ---")
        for t in ts:
            print(f"     {t}")
        print(f"     total U = {U:.4f}   rm_bound = {bound:.4f}")
        if U > 1.0:
            print("     EDF : INFEASIBLE (U > 1)")
            print("     RM  : INFEASIBLE (U > 1)")
            return
        print(f"     EDF verdict : {'SCHEDULABLE' if U <= 1.0 else 'NOT SCHEDULABLE'} (exact, U<=1)")
        if U <= bound:
            print(f"     RM  verdict : SCHEDULABLE by Liu & Layland bound "
                  f"(no RTA needed)")
        else:
            print("     RM bound fails -> falling back to exact RTA:")
            ok_all = True
            for t, R, it, ok in rta(ts, order):
                ok_all &= ok
                print(f"        {t.name}: R={R:>3}  D={t.D:>3}  "
                      f"({'ok' if ok else 'MISS'}, {it} iterations)")
            print(f"     RM  verdict : {'SCHEDULABLE' if ok_all else 'NOT SCHEDULABLE'} (exact RTA)")

    setA = [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12)]
    setB = setA + [Task("t4", 1, 8)]
    setC = [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12), Task("t4", 3, 8)]
    setD = [Task("a", 1, 5, 5), Task("b", 2, 8, 3), Task("c", 1, 10, 10)]  # constrained
    setE = [Task(f"e{i}", 3, 10) for i in range(1, 9)]                     # 8 tasks
    setF = [Task("x", 5, 10), Task("y", 6, 10)]                            # U > 1

    for nm, s in (("Set A (Sec 2.2.2)", setA), ("Set B (Sec 2.3.2)", setB),
                  ("Set C (RM-bound fails, RTA rescues)", setC),
                  ("Set D (constrained deadlines)", setD),
                  ("Set E (8 tasks, bound -> ln2)", setE),
                  ("Set F (U > 1, infeasible)", setF)):
        analyse(s, nm)

    # comparison chart for Set D (constrained deadlines, DM ordering)
    Hd = hyperperiod(setD)
    tl_dm, _ = simulate(setD, Hd, "DM")
    tl_edf, _ = simulate(setD, Hd, "EDF")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    draw_gantt(a1, tl_dm, [t.name for t in setD], "DM (RM with D-ordering)")
    draw_gantt(a2, tl_edf, [t.name for t in setD], "EDF")
    fig.suptitle("RM/DM vs EDF comparison - constrained deadline set")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_tool_compare.png", dpi=130); plt.close(fig)

    # ==================================================================
    # MINI PROJECT 2.3  -- partitioned multiprocessor scheduling
    # ==================================================================
    print("\n" + LINE)
    print("MINI PROJECT 2.3 - PARTITIONED MULTIPROCESSOR SCHEDULING (m = 4)")
    print(LINE)

    M = 4
    pset1 = [Task(f"b{i}", 1, 10) for i in range(1, 9)]                 # balanced
    pset2 = [Task("big1", 6, 11), Task("big2", 7, 13),
             Task("big3", 5, 9)] + [Task(f"s{i}", 1, 20) for i in range(1, 9)]
    pset3 = [Task("a", 6, 11), Task("b", 7, 13), Task("c", 5, 9),
             Task("d", 8, 15), Task("e", 9, 17)]                        # ADVERSARIAL
    pset4 = [Task("t1", 1, 4), Task("t2", 2, 6),
             Task("t3", 2, 12), Task("t4", 1, 8)]                       # worked example

    print(f"\n{'set':<12}{'heur':<6}{'cores':>7}{'success':>9}"
          f"{'total U':>10}{'U/m':>8}   per-core utilization")
    print("-" * 78)
    best_cfg = None
    for sname, s in (("balanced", pset1), ("skewed", pset2),
                     ("ADVERSARIAL", pset3), ("worked-ex", pset4)):
        for h in ("FF", "FFD", "BFD", "WF"):
            cores, loads, _ = partition(s, M, h)
            if cores is None:
                print(f"{sname:<12}{h:<6}{'-':>7}{'FAIL':>9}"
                      f"{total_utilization(s):>10.3f}{'':>8}   cannot fit in {M} cores")
                continue
            used = sum(1 for c in cores if c)
            tot = sum(loads)
            per = ", ".join(f"{l:.3f}" for l in loads)
            print(f"{sname:<12}{h:<6}{used:>7}{'ok':>9}{tot:>10.3f}"
                  f"{tot/M:>8.3f}   [{per}]")
            if sname == "worked-ex" and h == "FFD":
                best_cfg = cores
        print("-" * 78)

    print("\nADVERSARIAL analysis:")
    print("  pset3 = 5 tasks each with u_i slightly above 0.5 (total U = "
          f"{total_utilization(pset3):.3f} << m = {M}).")
    print("  rm_bound(2) = 0.828, so any two such tasks on one core are")
    print("  inadmissible -> each core holds at most ONE task -> 5 cores")
    print("  needed but only 4 available -> EVERY heuristic fails, despite")
    print("  total utilization being far below m (the bin-packing limitation")
    print("  of Section 2.8.1).")

    if best_cfg:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        pal = plt.cm.tab10.colors
        cmap, k = {}, 0
        for core in best_cfg:
            for t in core:
                if t.name not in cmap:
                    cmap[t.name] = pal[k % 10]; k += 1
        for i, core in enumerate(best_cfg):
            if not core:
                continue
            tl, _ = simulate(core, 24, "RM")
            segs = _runs(tl)
            for nm, runs in segs.items():
                col = "#eeeeee" if nm == "idle" else cmap[nm]
                ax.broken_barh(runs, (i - 0.4, 0.8), facecolors=col,
                               edgecolors="black", linewidth=0.4)
                if nm != "idle":
                    for (s0, w) in runs:
                        ax.text(s0 + w / 2, i, nm, ha="center", va="center",
                                fontsize=7)
        ax.set_yticks(range(len(best_cfg)))
        ax.set_yticklabels([f"core {i} (u={sum(t.U for t in c):.3f})"
                            for i, c in enumerate(best_cfg)])
        ax.set_ylim(-0.7, len(best_cfg) - 0.3)
        ax.set_xlabel("time (ticks)")
        ax.set_title("Per-core RM schedule of the 4-task set (FFD, m=4)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(f"{FIG}/fig_partition.png", dpi=130)
        plt.close(fig)

    print("\nAll figures written to ./figs/ :")
    for f in sorted(os.listdir(FIG)):
        print("   ", f)


if __name__ == "__main__":
    main()