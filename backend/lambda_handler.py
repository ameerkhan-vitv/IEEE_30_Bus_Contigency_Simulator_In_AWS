# lambda_handler.py
# AWS entry points for the grid simulator.
#   api_handler      - behind API Gateway (HTTP API): /solve, /n1, /scenarios, ...
#   nightly_handler  - triggered by EventBridge: runs the N-1 sweep, saves a report to S3,
#                      caches it in DynamoDB and emails the worst cases through SNS.
#
# Runs locally too: if TABLE_NAME is not set, an in-memory dict replaces DynamoDB
# (see local_test.py), so you can test everything without an AWS account.
import json
import math
import os
import re
import time
from decimal import Decimal

import numpy as np

from grid_data import IEEE_30_LINES, GEN_PMAX
from grid_engine import solve_system, run_n1_sweep, severity_index

TABLE_NAME = os.environ.get("TABLE_NAME")
TOPIC_ARN = os.environ.get("TOPIC_ARN")
REPORT_BUCKET = os.environ.get("REPORT_BUCKET")

ENGINE_VERSION = "1"            # change this to throw away old cached N-1 results
CACHE_TTL = 7 * 24 * 3600       # cached N-1 results live for a week
MAX_SCENARIOS_PER_USER = 50
VALID_LINES = {frozenset(l) for l in IEEE_30_LINES}
CLIENT_ID_RE = re.compile(r"[A-Za-z0-9-]{16,64}")   # random id made by each browser


# --------------------------------------------------------------------------
# JSON helpers (numpy / complex / Decimal are not JSON-serialisable by default)
# --------------------------------------------------------------------------
def _default(o):
    if isinstance(o, Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"cannot serialise {type(o)}")


def to_json(res):
    # turn solve_system() output into plain JSON: drop the networkx graph,
    # turn complex voltages into magnitude + angle
    islands = []
    for i in res["islands"]:
        d = {k: v for k, v in i.items() if k != "V"}
        d["V"] = {str(b): {"vm": abs(v), "va": math.degrees(np.angle(v))}
                  for b, v in (i["V"] or {}).items()}
        islands.append(d)
    return dict(islands=islands, bus=res["bus"], lines=res["lines"], totals=res["totals"],
                violations=res["violations"], broken=res["broken"],
                load_scale=res["load_scale"], tripped_gens=res["tripped_gens"])


def resp(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, default=_default)}


def emit_metric(name, value, unit="None"):
    # CloudWatch Embedded Metric Format: printing this line creates a custom metric for free
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000),
                 "CloudWatchMetrics": [{"Namespace": "GridSim", "Dimensions": [[]],
                                        "Metrics": [{"Name": name, "Unit": unit}]}]},
        name: value}))


# --------------------------------------------------------------------------
# DynamoDB (single table: pk + sk). In-memory fallback for local testing.
# --------------------------------------------------------------------------
_LOCAL_DB = {}
_table = None


def _get_table():
    global _table
    if _table is None:
        import boto3
        _table = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _table


def _to_ddb(obj):
    # DynamoDB wants Decimal instead of float and string map keys
    return json.loads(json.dumps(obj, default=_default), parse_float=Decimal)


def db_put(item):
    item = _to_ddb(item)
    if TABLE_NAME:
        _get_table().put_item(Item=item)
    else:
        _LOCAL_DB[(item["pk"], item["sk"])] = item


def db_get(pk, sk):
    if TABLE_NAME:
        return _get_table().get_item(Key={"pk": pk, "sk": sk}).get("Item")
    return _LOCAL_DB.get((pk, sk))


def db_query(pk, prefix):
    if TABLE_NAME:
        from boto3.dynamodb.conditions import Key
        return _get_table().query(
            KeyConditionExpression=Key("pk").eq(pk) & Key("sk").begins_with(prefix),
            ScanIndexForward=False).get("Items", [])
    items = [v for (p, s), v in _LOCAL_DB.items() if p == pk and s.startswith(prefix)]
    return sorted(items, key=lambda i: i["sk"], reverse=True)


def db_delete(pk, sk):
    if TABLE_NAME:
        _get_table().delete_item(Key={"pk": pk, "sk": sk})
    else:
        _LOCAL_DB.pop((pk, sk), None)


# --------------------------------------------------------------------------
# Input checking
# --------------------------------------------------------------------------
class BadRequest(Exception):
    pass


def parse_params(d):
    lines = []
    for l in d.get("lines", []):
        if not (isinstance(l, (list, tuple)) and len(l) == 2):
            raise BadRequest("each line must look like [from_bus, to_bus]")
        try:
            a, b = int(l[0]), int(l[1])
        except (TypeError, ValueError):
            raise BadRequest("bus numbers must be integers")
        if frozenset((a, b)) not in VALID_LINES:
            raise BadRequest(f"line {a}-{b} does not exist")
        lines.append((a, b))
    gens = []
    for g in d.get("gens", []):
        try:
            g = int(g)
        except (TypeError, ValueError):
            raise BadRequest("generator buses must be integers")
        if g not in GEN_PMAX:
            raise BadRequest(f"bus {g} has no generator")
        gens.append(g)
    try:
        scale = float(d.get("scale", 100))
    except (TypeError, ValueError):
        raise BadRequest("scale must be a number (percent)")
    if not 10 <= scale <= 200:
        raise BadRequest("scale must be between 10 and 200 (percent)")
    return lines, scale / 100.0, gens


# --------------------------------------------------------------------------
# Actual work
# --------------------------------------------------------------------------
def do_solve(d):
    lines, scale, gens = parse_params(d)
    t0 = time.time()
    res = solve_system(lines, scale, gens)
    ms = (time.time() - t0) * 1000
    emit_metric("SolveTimeMs", round(ms, 1), "Milliseconds")
    out = to_json(res)
    out["severity"] = severity_index(res)
    out["solve_ms"] = round(ms, 1)
    return out


def n1_rows(scale=1.0, gens=(), use_cache=True):
    # the N-1 sweep is 41 power flows, so keep the result in DynamoDB
    key = f"N1#{ENGINE_VERSION}#{int(round(scale * 100))}#{','.join(map(str, sorted(gens)))}"
    if use_cache:
        hit = db_get("CACHE", key)
        if hit and float(hit.get("ttl", 0)) > time.time():
            return json.loads(json.dumps(hit["rows"], default=_default)), True
    t0 = time.time()
    rows = run_n1_sweep(scale, gens)
    emit_metric("N1SweepTimeMs", round((time.time() - t0) * 1000, 1), "Milliseconds")
    rows = json.loads(json.dumps(rows, default=_default))
    db_put({"pk": "CACHE", "sk": key, "rows": rows, "ttl": int(time.time()) + CACHE_TTL})
    return rows, False


def _summary(res):
    t = res["totals"]
    return dict(islands=len(res["islands"]), load_lost=round(t["demand"] - t["served"], 2),
                losses=round(t["losses"], 2), overloads=len(res["violations"]["overload"]),
                voltage_violations=len(res["violations"]["voltage"]))


# --------------------------------------------------------------------------
# API Gateway entry point
# --------------------------------------------------------------------------
def api_handler(event, context):
    try:
        http = event["requestContext"]["http"]
        method, path = http["method"], event["rawPath"].rstrip("/") or "/"
        body = json.loads(event["body"]) if event.get("body") else {}
        qs = event.get("queryStringParameters") or {}
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        user = headers.get("x-client-id", "")
        if not CLIENT_ID_RE.fullmatch(user):
            user = None

        if method == "GET" and path == "/health":
            res = solve_system()
            ok = 17.3 <= res["totals"]["losses"] <= 17.9
            return resp(200 if ok else 500, {"ok": ok, "base_losses_mw": round(res["totals"]["losses"], 2)})

        if method == "GET" and path == "/demo":
            from grid_tests import SCENARIOS
            return resp(200, SCENARIOS)

        if method == "POST" and path == "/solve":
            return resp(200, do_solve(body))

        if method == "GET" and path == "/n1":
            _, scale, gens = parse_params({"scale": qs.get("scale", 100),
                                           "gens": [g for g in qs.get("gens", "").split(",") if g]})
            rows, cached = n1_rows(scale, gens)
            return resp(200, {"cached": cached, "rows": rows})

        if method == "GET" and path == "/validate":
            from grid_tests import run_validation
            checks = run_validation()
            return resp(200, {"passed": sum(1 for _, p, _ in checks if p), "total": len(checks),
                              "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks]})

        # saved scenarios are kept per browser, identified by the X-Client-Id header
        if not user:
            return resp(401, {"error": "missing or invalid X-Client-Id header"})

        if path == "/scenarios" and method == "GET":
            items = db_query(f"USER#{user}", "SCEN#")
            return resp(200, [dict(i, id=i["sk"].split("#", 1)[1]) for i in items])

        if path == "/scenarios" and method == "POST":
            lines, scale, gens = parse_params(body)
            if len(db_query(f"USER#{user}", "SCEN#")) >= MAX_SCENARIOS_PER_USER:
                raise BadRequest(f"you can save at most {MAX_SCENARIOS_PER_USER} scenarios")
            sid = f"{int(time.time() * 1000):013d}"
            item = dict(pk=f"USER#{user}", sk=f"SCEN#{sid}", name=str(body.get("name", "Untitled"))[:60],
                        lines=[list(l) for l in lines], gens=gens, scale=scale * 100,
                        summary=_summary(solve_system(lines, scale, gens)))
            db_put(item)
            return resp(201, dict(item, id=sid))

        if path.startswith("/scenarios/") and method == "DELETE":
            db_delete(f"USER#{user}", "SCEN#" + path.rsplit("/", 1)[1])
            return resp(200, {"deleted": True})

        return resp(404, {"error": "not found"})

    except BadRequest as e:
        return resp(400, {"error": str(e)})
    except json.JSONDecodeError:
        return resp(400, {"error": "body is not valid JSON"})
    except Exception as e:                       # shows up in CloudWatch Logs + the error alarm
        print(f"ERROR {type(e).__name__}: {e}")
        raise


# --------------------------------------------------------------------------
# EventBridge (nightly) entry point
# --------------------------------------------------------------------------
def nightly_handler(event, context):
    rows, _ = n1_rows(1.0, [], use_cache=False)       # always recompute, refreshes the cache
    worst = rows[:3]
    emit_metric("WorstSeverity", worst[0]["severity"])
    today = time.strftime("%Y-%m-%d", time.gmtime())
    lines = [f"  {i}. line {r['line'][0]}-{r['line'][1]}: {r['lost']:.1f} MW lost, "
             f"{r['overloads']} overload(s), min V {r['min_v']:.3f} pu"
             for i, r in enumerate(worst, 1)]
    message = f"Nightly N-1 check ({today}) - 3 worst single-line outages:\n" + "\n".join(lines)

    if REPORT_BUCKET:
        import boto3
        boto3.client("s3").put_object(Bucket=REPORT_BUCKET, Key=f"reports/n1-{today}.json",
                                      Body=json.dumps(rows, default=_default),
                                      ContentType="application/json")
    if TOPIC_ARN:
        import boto3
        boto3.client("sns").publish(TopicArn=TOPIC_ARN, Subject=f"Grid N-1 report {today}",
                                    Message=message)
    print(message)
    return {"ok": True, "worst": worst[0]["line"]}
