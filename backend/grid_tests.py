# grid_tests.py - demo scenarios and the self-check list (no tkinter in here)
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

from grid_data import IEEE_30_LINES, GEN_PMAX, BUS_LOADS, V_MIN, V_MAX
from grid_engine import solve_system, build_ybus, run_n1_sweep

# Demo scenarios for the drop-down. The auto-demo plays them in this order.
# "tab" is the chart tab that shows the effect best.
SCENARIOS = [
    dict(name="1. Normal day - nothing wrong", lines=[], gens=[], scale=100, tab="voltage",
         desc="Nothing is cut. All voltages are fine, no line is overloaded, and about 17.6 MW "
              "is lost as heat in the wires (the published IEEE-30 value)."),
    dict(name="2. A dead-end line fails (25-26)", lines=[(25, 26)], gens=[], scale=100, tab="balance",
         desc="Bus 26 is only connected through line 25-26. Cutting it leaves bus 26 alone with no "
              "generator, so it blacks out (3.5 MW). The rest of the grid carries on normally."),
    dict(name="3. An important line fails (2-5)", lines=[(2, 5)], gens=[], scale=100, tab="loading",
         desc="This line carries a lot of power towards bus 5. The grid stays connected, but power "
              "has to take other routes, so some lines overload and some voltages drop."),
    dict(name="4. Main line from generator 1 fails (1-2)", lines=[(1, 2)], gens=[], scale=100, tab="loading",
         desc="Generator 1 loses its strongest connection. The remaining line can't carry "
              "everything, so the system cuts some load until it can keep running."),
    dict(name="5. Generator 1 cut off (lines 1-2 and 1-3)", lines=[(1, 2), (1, 3)], gens=[], scale=100,
         tab="balance",
         desc="The big plant at bus 1 is cut off from all customers. The rest of the grid has only "
              "235 MW of its own generators, so it runs them flat out and cuts the rest of the load."),
    dict(name="6. Biggest generator shuts down (bus 1)", lines=[], gens=[1], scale=100, tab="balance",
         desc="Generator 1 trips. Bus 2 takes over as the balancing bus, the other generators go to "
              "maximum, and whatever is still missing is cut."),
    dict(name="7. Very high demand (130%)", lines=[], gens=[], scale=130, tab="voltage",
         desc="Nothing is cut, but everyone uses 30% more. Voltages sag below the limit at the far "
              "end of the grid and the wire losses roughly double."),
    dict(name="8. Big fault - grid splits in two", lines=[(2, 5), (2, 6), (4, 6), (6, 7)], gens=[],
         scale=100, tab="balance",
         desc="Four lines near bus 6 fail together and the grid splits into two islands. Each island "
              "balances itself, with its own balancing bus and its own amount of load cut."),
    dict(name="9. Line 1-2 fails at high demand (120%)", lines=[(1, 2)], gens=[], scale=120, tab="balance",
         desc="The same fault as scenario 4, but at higher demand. Far more load has to be cut."),
]


def summarize(res):
    # the one-line result shown under the description
    t = res["totals"]
    live = [b["vm"] for b in res["bus"].values() if b["energized"]]
    return (f"Result: {len(res['islands'])} island(s) | {t['demand'] - t['served']:.1f} MW load lost | "
            f"{len(res['violations']['overload'])} line(s) overloaded | "
            f"{len(res['violations']['voltage'])} bus(es) outside voltage limits | "
            f"lowest voltage {min(live) if live else 0:.3f} pu | wire losses {t['losses']:.1f} MW")


# --------------------------------------------------------------------------
# Self-checks: each one records (name, passed, detail)
# --------------------------------------------------------------------------
def _balance_ok(res, tol=0.05):
    t = res["totals"]
    return abs(t["gen"] - t["served"] - t["losses"]) < tol


def _kcl_ok(res, tol=1e-3):
    # at every live bus: generation - load served = power flowing out on the lines
    worst = 0.0
    for b, d in res["bus"].items():
        if not d["energized"]:
            continue
        out = sum(l["p_from"] for l in res["lines"] if l["energized"] and l["f"] == b) + \
              sum(l["p_to"] for l in res["lines"] if l["energized"] and l["t"] == b)
        worst = max(worst, abs((d["pg"] - d["served"]) - out))
    return worst < tol, worst


def run_validation():
    checks = []

    def add(name, passed, detail):
        checks.append((name, bool(passed), detail))

    base = solve_system()
    t = base["totals"]
    add("Base case solves", all(i["converged"] for i in base["islands"]),
        f"{len(base['islands'])} island, status: {base['islands'][0]['status']}")
    add("Base-case losses match the published value (~17.6 MW)", 17.3 <= t["losses"] <= 17.9,
        f"computed {t['losses']:.2f} MW")
    add("Generator 1 output is about 261 MW", 259.5 <= base["bus"][1]["pg"] <= 262.5,
        f"computed {base['bus'][1]['pg']:.1f} MW")
    add("Total demand is 283.4 MW", abs(sum(BUS_LOADS.values()) - 283.4) < 1e-6,
        f"{sum(BUS_LOADS.values()):.1f} MW")
    vmin = min(d["vm"] for d in base["bus"].values())
    add("Base-case voltages are inside the limits", V_MIN <= vmin and not base["violations"]["voltage"],
        f"lowest {vmin:.3f} pu (limit {V_MIN})")
    add("No line is overloaded in the base case", not base["violations"]["overload"],
        f"max loading {max(l['loading'] for l in base['lines']):.0f}%")
    add("Bus 3 and bus 30 voltages match the published values",
        abs(base["bus"][3]["vm"] - 1.021) < 0.003 and abs(base["bus"][30]["vm"] - 0.991) < 0.003,
        f"V3={base['bus'][3]['vm']:.3f}, V30={base['bus'][30]['vm']:.3f}")

    Y = build_ybus()
    add("Ybus matrix is symmetric", np.allclose(Y, Y.T), "max |Y - Y^T| = "
        f"{np.max(np.abs(Y - Y.T)):.1e}")

    ok, worst = _kcl_ok(base)
    add("Power in = power out at every bus (base case)", ok, f"worst mismatch {worst:.2e} MW")

    s = solve_system([(25, 26)])
    add("Cutting 25-26 gives exactly 2 islands", len(s["islands"]) == 2, f"{len(s['islands'])} islands")
    add("Bus 26 blacks out, 3.5 MW lost",
        any(i["code"] == "dead" and i["buses"] == [26] for i in s["islands"])
        and abs(s["totals"]["demand"] - s["totals"]["served"] - 3.5) < 0.05,
        f"lost {s['totals']['demand'] - s['totals']['served']:.1f} MW")

    G = solve_system([(1, 2), (1, 3)])
    if nx is not None:
        nx_cnt = nx.number_connected_components(G["G"])
        add("Island count matches networkx", nx_cnt == len(G["islands"]),
            f"{nx_cnt} vs {len(G['islands'])}")
    isl = max(G["islands"], key=lambda i: i["demand"])
    add("Cut-off grid runs its generators at their 235 MW maximum",
        abs(isl["gen"] - 235.0) < 0.5, f"generation {isl['gen']:.1f} MW")
    add("Cut-off grid: some load cut, and served + cut = demand",
        isl["shed"] > 1 and abs(isl["served"] + isl["shed"] - isl["demand"]) < 1e-6,
        f"shed {isl['shed']:.1f} MW")

    g1 = solve_system([], 1.0, [1])
    add("Generator 1 trip: bus 2 takes over as balancing bus", g1["islands"][0]["slack"] == 2,
        f"slack = bus {g1['islands'][0]['slack']}")

    for name, args in (("line 2-5 out", ([(2, 5)],)), ("grid split", ([(2, 5), (2, 6), (4, 6), (6, 7)],)),
                       ("130% load", ([], 1.3)), ("gen 1 trip", ([], 1.0, [1]))):
        r = solve_system(*args)
        k_ok, k_w = _kcl_ok(r)
        add(f"Generation = load served + losses ({name})", _balance_ok(r), f"gap {abs(r['totals']['gen'] - r['totals']['served'] - r['totals']['losses']):.3f} MW")
        add(f"Power in = power out at every bus ({name})", k_ok, f"worst mismatch {k_w:.2e} MW")

    heavy = solve_system([], 1.3)
    add("130% demand: losses clearly higher than normal", heavy["totals"]["losses"] > 1.5 * t["losses"],
        f"{heavy['totals']['losses']:.1f} MW vs {t['losses']:.1f} MW")

    rows = run_n1_sweep()
    add("N-1 check covers all 41 lines", len(rows) == len(IEEE_30_LINES), f"{len(rows)} cases")
    sev = [r["severity"] for r in rows]
    add("N-1 results are sorted worst first", sev == sorted(sev, reverse=True),
        f"worst case: line {rows[0]['line'][0]}-{rows[0]['line'][1]}")

    for sc in SCENARIOS:
        r = solve_system(sc["lines"], sc["scale"] / 100.0, sc["gens"])
        add(f"Demo scenario runs: {sc['name']}",
            all(i["code"] != "collapse" for i in r["islands"]) and _balance_ok(r, 0.1),
            summarize(r)[8:])
    return checks
