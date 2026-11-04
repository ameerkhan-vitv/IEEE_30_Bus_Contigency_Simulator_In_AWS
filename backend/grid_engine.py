# grid_engine.py
# The calculations behind the simulator:
#   - find which parts of the grid are still connected (islands)
#   - build the Ybus matrix
#   - solve the power flow (Newton-Raphson)
#   - balance each island: turn generators up/down, shed load if needed
#   - run the one-line-at-a-time (N-1) outage check
import numpy as np

try:                      # networkx is optional: only used to build the graph object for the desktop GUI
    import networkx as nx
except ImportError:
    nx = None

from grid_data import (
    BASE_MVA, V_MIN, V_MAX, BRANCHES, IEEE_30_LINES, LINE_RATING_MVA,
    BUS_LOADS, BUS_LOADS_Q, BUS_SHUNT_MVAR, BUS_GENERATION,
    GEN_PMAX, GEN_QMIN, GEN_QMAX, GEN_VSET,
)

NBUS = 30
TOL = 1e-6          # how close to zero the mismatch must get
MAX_ITER = 25


def broken_indices(broken_lines):
    # turn (from, to) pairs into line numbers, in either direction
    wanted = {frozenset(l) for l in broken_lines}
    return {i for i, (a, b) in enumerate(IEEE_30_LINES) if frozenset((a, b)) in wanted}


def _components(out):
    # which buses are still connected to each other (plain search, no networkx needed)
    adj = {b: set() for b in range(1, NBUS + 1)}
    for k, (a, b) in enumerate(IEEE_30_LINES):
        if k not in out:
            adj[a].add(b)
            adj[b].add(a)
    seen, comps = set(), []
    for start in range(1, NBUS + 1):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _line_y(k):
    # the four admittance terms of line k (handles transformer tap too)
    f, t, r, x, b, tap = BRANCHES[k]
    ys = 1.0 / complex(r, x)
    half_charge = 1j * b / 2.0
    return (ys + half_charge) / tap ** 2, -ys / tap, -ys / tap, ys + half_charge


def build_ybus(broken_lines=(), buses=None):
    out = broken_indices(broken_lines)
    Y = np.zeros((NBUS, NBUS), dtype=complex)
    for k, line in enumerate(BRANCHES):
        if k in out:
            continue
        f, t = line[0] - 1, line[1] - 1
        yff, yft, ytf, ytt = _line_y(k)
        Y[f, f] += yff
        Y[f, t] += yft
        Y[t, f] += ytf
        Y[t, t] += ytt
    for b, mvar in BUS_SHUNT_MVAR.items():
        Y[b - 1, b - 1] += 1j * mvar / BASE_MVA
    if buses is not None:
        idx = [b - 1 for b in buses]
        Y = Y[np.ix_(idx, idx)]
    return Y


def _newton_raphson(Y, S, V0, pv, pq):
    # standard polar Newton-Raphson. pv / pq are local bus indexes.
    V = V0.astype(complex).copy()
    pq = np.asarray(pq, dtype=int)
    pvpq = np.r_[pv, pq].astype(int)
    if len(pvpq) == 0:
        return V, True
    for it in range(MAX_ITER + 1):
        I = Y @ V
        mismatch = V * np.conj(I) - S
        F = np.r_[mismatch[pvpq].real, mismatch[pq].imag]
        if np.max(np.abs(F)) < TOL:
            return V, True
        if it == MAX_ITER:
            break
        dV = np.diag(V)
        dI = np.diag(I)
        dVn = np.diag(V / np.abs(V))
        dS_dVm = dV @ np.conj(Y @ dVn) + np.conj(dI) @ dVn
        dS_dVa = 1j * dV @ np.conj(dI - Y @ dV)
        J = np.block([
            [dS_dVa[np.ix_(pvpq, pvpq)].real, dS_dVm[np.ix_(pvpq, pq)].real],
            [dS_dVa[np.ix_(pq, pvpq)].imag,   dS_dVm[np.ix_(pq, pq)].imag],
        ])
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return V, False
        mag, ang = np.abs(V), np.angle(V)
        ang[pvpq] += step[:len(pvpq)]
        mag[pq] += step[len(pvpq):]
        if not np.all(np.isfinite(mag)) or mag.max() > 3 or mag.min() < 0.05:
            return V, False          # blew up
        V = mag * np.exp(1j * ang)
    return V, False


def _power_flow(buses, active, Pd, Qd, Pg, slack, gens):
    # Power flow for one island.
    # Returns (voltages, converged, slack P in MW, Q of each generator in MVAr)
    n = len(buses)
    pos = {b: i for i, b in enumerate(buses)}

    if n == 1:      # a single bus with a generator, nothing to solve
        b = buses[0]
        q = Qd[b] - BUS_SHUNT_MVAR.get(b, 0.0) * GEN_VSET[b] ** 2
        return {b: complex(GEN_VSET[b], 0)}, True, Pd[b], {b: q}

    Y = np.zeros((n, n), dtype=complex)
    for k in active:
        i, j = pos[BRANCHES[k][0]], pos[BRANCHES[k][1]]
        yff, yft, ytf, ytt = _line_y(k)
        Y[i, i] += yff
        Y[i, j] += yft
        Y[j, i] += ytf
        Y[j, j] += ytt
    for b in buses:
        if b in BUS_SHUNT_MVAR:
            Y[pos[b], pos[b]] += 1j * BUS_SHUNT_MVAR[b] / BASE_MVA

    pv_gens = [g for g in gens if g != slack]
    q_stuck = {}        # generators that hit a Q limit and are now held at it

    for _ in range(12):
        S = np.zeros(n, dtype=complex)
        for b in buses:
            S[pos[b]] = complex(-Pd[b], -Qd[b]) / BASE_MVA
        for g in pv_gens:
            S[pos[g]] += Pg[g] / BASE_MVA
            if g in q_stuck:
                S[pos[g]] += 1j * q_stuck[g] / BASE_MVA

        V0 = np.ones(n, dtype=complex)
        for g in gens:
            if g == slack or g not in q_stuck:
                V0[pos[g]] = GEN_VSET[g]
        pv = [pos[g] for g in pv_gens if g not in q_stuck]
        pq = [i for i in range(n) if i != pos[slack] and i not in pv]

        V, ok = _newton_raphson(Y, S, V0, np.array(pv, dtype=int), np.array(pq, dtype=int))
        if not ok:
            return None, False, 0.0, {}

        injected = V * np.conj(Y @ V)
        hit_limit = False
        for g in pv_gens:
            if g in q_stuck:
                continue
            q = injected[pos[g]].imag * BASE_MVA + Qd[g]
            if q > GEN_QMAX[g] + 1e-6:
                q_stuck[g] = GEN_QMAX[g]
                hit_limit = True
            elif q < GEN_QMIN[g] - 1e-6:
                q_stuck[g] = GEN_QMIN[g]
                hit_limit = True
        if not hit_limit:
            break

    p_slack = injected[pos[slack]].real * BASE_MVA + Pd[slack]
    q_out = {g: (q_stuck[g] if g in q_stuck else injected[pos[g]].imag * BASE_MVA + Qd[g])
             for g in gens}
    return {b: V[pos[b]] for b in buses}, True, p_slack, q_out


def _solve_island(buses, active, load_scale, tripped):
    gens = [g for g in GEN_PMAX if g in buses and g not in tripped]
    P0 = {b: BUS_LOADS.get(b, 0.0) * load_scale for b in buses}
    Q0 = {b: BUS_LOADS_Q.get(b, 0.0) * load_scale for b in buses}
    demand = sum(P0.values())

    isl = dict(buses=sorted(buses), demand=demand, gens=gens, slack=None, served=0.0,
               shed=demand, gen=0.0, losses=0.0, curtailed=0.0, redispatched=False,
               V=None, Qg={}, Pg={}, shed_frac=1.0, converged=True)

    if not gens:
        isl["code"] = "dead"
        isl["status"] = ("Blackout - no generator in this area" if demand > 0
                         else "No supply - empty area")
        return isl

    # the biggest generator carries the imbalance (bus 1 when it is in the island)
    slack = 1 if 1 in gens else max(gens, key=lambda g: GEN_PMAX[g])
    others = [g for g in gens if g != slack]
    planned = {g: min(BUS_GENERATION.get(g, 0.0), GEN_PMAX[g]) for g in others}
    Pg = dict(planned)
    shed_frac = 0.0
    result = None

    for _ in range(60):
        Pd = {b: P0[b] * (1 - shed_frac) for b in buses}
        Qd = {b: Q0[b] * (1 - shed_frac) for b in buses}
        V, ok, p_slack, Qg = _power_flow(buses, active, Pd, Qd, Pg, slack, gens)

        if not ok:
            # no solution found - drop 10% more load and try again
            if shed_frac < 0.6 and demand > 0:
                shed_frac += 0.1
                continue
            isl.update(code="collapse", converged=False,
                       status="Voltage collapse - no stable solution")
            return isl

        too_much = p_slack - GEN_PMAX[slack]
        if too_much > 0.01:
            # slack generator is overloaded: turn the others up first
            need = too_much
            for g in sorted(others, key=lambda g: GEN_PMAX[g] - Pg[g], reverse=True):
                add = min(GEN_PMAX[g] - Pg[g], need)
                Pg[g] += add
                need -= add
            if need > 0.01:         # nothing left to turn up, so shed load
                left = demand * (1 - shed_frac)
                if left > 0:
                    shed_frac = min(1.0, shed_frac + need / left * (1 - shed_frac) * 1.02)
                else:
                    shed_frac = 1.0
            continue

        if p_slack < -0.01 and others:
            # too much generation: turn the others down
            total = sum(Pg.values())
            if total <= 0:
                break
            cut = min(total, -p_slack)
            for g in others:
                Pg[g] -= cut * Pg[g] / total
            continue

        result = (V, p_slack, Qg)
        break

    # Shedding tends to overshoot a little (losses drop once load is shed),
    # so squeeze it back with a simple halving search.
    if result is not None and shed_frac > 1e-4 and all(Pg[g] >= GEN_PMAX[g] - 0.01 for g in others):
        low, high, best = 0.0, shed_frac, result
        for _ in range(25):
            if high - low < 1e-4:
                break
            mid = (low + high) / 2
            Pd = {b: P0[b] * (1 - mid) for b in buses}
            Qd = {b: Q0[b] * (1 - mid) for b in buses}
            V, ok, p_slack, Qg = _power_flow(buses, active, Pd, Qd, Pg, slack, gens)
            if ok and p_slack <= GEN_PMAX[slack] + 0.01:
                high, best = mid, (V, p_slack, Qg)
            else:
                low = mid
        shed_frac, result = high, best

    if result is None:
        isl.update(code="collapse", converged=False,
                   status="Voltage collapse - no stable solution")
        return isl

    V, p_slack, Qg = result
    Pg_all = dict(Pg)
    Pg_all[slack] = p_slack
    gen_total = sum(Pg_all.values())
    served = demand * (1 - shed_frac)
    curtailed = sum(max(0.0, planned[g] - Pg[g]) for g in others)
    turned_up = any(Pg[g] > planned[g] + 0.01 for g in others)

    isl.update(slack=slack, V=V, Qg=Qg, Pg=Pg_all, gen=gen_total, served=served,
               shed=demand - served, losses=gen_total - served, shed_frac=shed_frac,
               curtailed=curtailed, redispatched=turned_up)

    if isl["shed"] > 0.05:
        isl["code"], isl["status"] = "shed", "Shedding load to keep the island running"
    elif turned_up:
        isl["code"], isl["status"] = "redispatch", "Stable - local generators turned up"
    elif curtailed > 0.05:
        isl["code"], isl["status"] = "curtailed", "Stable - extra generation turned down"
    else:
        isl["code"], isl["status"] = "normal", "Running normally"
    return isl


def solve_system(broken_lines=(), load_scale=1.0, tripped_gens=()):
    # Solve the whole grid for a given set of cut lines, load level and tripped generators.
    out = broken_indices(broken_lines)
    tripped = set(tripped_gens)

    G = None
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(range(1, NBUS + 1))
        G.add_edges_from(e for k, e in enumerate(IEEE_30_LINES) if k not in out)
    parts = sorted(_components(out), key=lambda c: (-len(c), c[0]))

    islands = []
    Vall = np.zeros(NBUS, dtype=complex)
    bus = {b: dict(vm=0.0, va=0.0, pd=BUS_LOADS.get(b, 0.0) * load_scale, served=0.0,
                   pg=0.0, qg=0.0, energized=False, island=0)
           for b in range(1, NBUS + 1)}

    for n, comp in enumerate(parts, start=1):
        members = set(comp)
        active = [k for k, line in enumerate(BRANCHES)
                  if k not in out and line[0] in members and line[1] in members]
        isl = _solve_island(comp, active, load_scale, tripped)
        isl["id"] = n
        islands.append(isl)
        for b in comp:
            bus[b]["island"] = n
        if isl["V"]:
            for b, v in isl["V"].items():
                Vall[b - 1] = v
                bus[b].update(vm=abs(v), va=float(np.degrees(np.angle(v))), energized=True,
                              served=bus[b]["pd"] * (1 - isl["shed_frac"]))
            for g, p in isl["Pg"].items():
                bus[g]["pg"] = p
            for g, q in isl["Qg"].items():
                bus[g]["qg"] = q

    lines = []
    for k, line in enumerate(BRANCHES):
        f, t = line[0], line[1]
        rec = dict(idx=k, f=f, t=t, in_service=k not in out, p_from=0.0, q_from=0.0,
                   p_to=0.0, q_to=0.0, loss=0.0, rating=LINE_RATING_MVA[k],
                   loading=0.0, energized=False)
        if rec["in_service"] and bus[f]["energized"] and bus[t]["energized"]:
            yff, yft, ytf, ytt = _line_y(k)
            Vf, Vt = Vall[f - 1], Vall[t - 1]
            Sf = Vf * np.conj(yff * Vf + yft * Vt) * BASE_MVA
            St = Vt * np.conj(ytf * Vf + ytt * Vt) * BASE_MVA
            rec.update(p_from=Sf.real, q_from=Sf.imag, p_to=St.real, q_to=St.imag,
                       loss=(Sf + St).real, energized=True,
                       loading=100.0 * max(abs(Sf), abs(St)) / rec["rating"])
        lines.append(rec)

    bad_voltage = [(b, d["vm"]) for b, d in bus.items()
                   if d["energized"] and (d["vm"] < V_MIN or d["vm"] > V_MAX)]
    overloaded = [(l["idx"], l["loading"]) for l in lines if l["loading"] > 100.0]

    totals = dict(
        demand=sum(i["demand"] for i in islands),
        served=sum(i["served"] for i in islands),
        shed=sum(i["demand"] - i["served"] for i in islands),
        gen=sum(i["gen"] for i in islands),
        losses=sum(l["loss"] for l in lines),
    )
    return dict(G=G, islands=islands, bus=bus, lines=lines, totals=totals,
                violations=dict(voltage=bad_voltage, overload=overloaded),
                broken=list(broken_lines), load_scale=load_scale,
                tripped_gens=sorted(tripped))


def severity_index(res):
    # One number to rank outages: MW of load lost, plus 0.5 for every % of overload,
    # plus 100 for every 1.0 pu of voltage outside the limits.
    lost = res["totals"]["demand"] - res["totals"]["served"]
    over = sum(max(0.0, x - 100.0) for _, x in res["violations"]["overload"])
    volt = sum(max(0.0, V_MIN - v) + max(0.0, v - V_MAX) for _, v in res["violations"]["voltage"])
    return lost + 0.5 * over + 100.0 * volt


def run_n1_sweep(load_scale=1.0, tripped_gens=()):
    # Cut each line one at a time and rank the results, worst first.
    rows = []
    for f, t in IEEE_30_LINES:
        res = solve_system([(f, t)], load_scale, tripped_gens)
        live = [b["vm"] for b in res["bus"].values() if b["energized"]]
        rows.append(dict(
            line=(f, t),
            islands=len(res["islands"]),
            lost=res["totals"]["demand"] - res["totals"]["served"],
            overloads=len(res["violations"]["overload"]),
            max_loading=max(l["loading"] for l in res["lines"]),
            min_v=min(live) if live else 0.0,
            volt_viol=len(res["violations"]["voltage"]),
            losses=res["totals"]["losses"],
            severity=severity_index(res),
        ))
    rows.sort(key=lambda r: r["severity"], reverse=True)
    return rows


def evaluate_network_contingency(broken_lines):
    # kept so the old version of the GUI still works
    res = solve_system(broken_lines)
    reports = [dict(id=i["id"], buses=i["buses"], gen=i["gen"], load=i["demand"],
                    status=i["status"]) for i in res["islands"]]
    return res["G"], reports
