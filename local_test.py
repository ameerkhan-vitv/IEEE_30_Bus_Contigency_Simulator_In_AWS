# local_test.py - test the Lambda code on your laptop, no AWS needed.
#   python local_test.py
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from lambda_handler import api_handler, nightly_handler   # noqa: E402


def call(method, path, body=None, qs=None, user=None):
    event = {"rawPath": path, "body": json.dumps(body) if body is not None else None,
             "queryStringParameters": qs,
             "headers": {"X-Client-Id": user} if user else {},
             "requestContext": {"http": {"method": method}}}
    r = api_handler(event, None)
    return r["statusCode"], json.loads(r["body"])


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({extra})" if extra else ""))
    return cond


ok = True
code, b = call("GET", "/health")
ok &= check("GET /health", code == 200 and b["ok"], b)

code, b = call("POST", "/solve", {"lines": [[1, 2]], "scale": 120})
ok &= check("POST /solve (line 1-2 out, 120% load)", code == 200 and b["totals"]["shed"] > 0,
            f"{b['totals']['shed']:.1f} MW shed, {b['solve_ms']} ms")
json.dumps(b)  # must be plain JSON

code, b = call("POST", "/solve", {"lines": [[1, 99]]})
ok &= check("bad line gives 400", code == 400, b)
code, b = call("POST", "/solve", {"scale": 500})
ok &= check("bad scale gives 400", code == 400, b)

code, b = call("GET", "/n1")
ok &= check("GET /n1 computes", code == 200 and not b["cached"] and len(b["rows"]) == 41)
code, b2 = call("GET", "/n1")
ok &= check("GET /n1 second time is cached", b2["cached"] and b2["rows"][0]["line"] == b["rows"][0]["line"])

code, b = call("GET", "/scenarios")
ok &= check("scenarios need a client id", code == 401)
code, b = call("POST", "/scenarios", {"name": "test", "lines": [[2, 5]], "scale": 100}, user="client-aaaaaaaaaaaa1")
ok &= check("save scenario", code == 201, b.get("summary"))
sid = b["id"]
code, b = call("GET", "/scenarios", user="client-aaaaaaaaaaaa1")
ok &= check("list scenarios", code == 200 and len(b) == 1)
code, b = call("GET", "/scenarios", user="client-bbbbbbbbbbbb2")
ok &= check("other user sees nothing", code == 200 and len(b) == 0)
code, b = call("DELETE", f"/scenarios/{sid}", user="client-aaaaaaaaaaaa1")
ok &= check("delete scenario", code == 200)

code, b = call("GET", "/validate")
ok &= check("self-checks all pass", code == 200 and b["passed"] == b["total"], f"{b['passed']}/{b['total']}")

r = nightly_handler({}, None)
ok &= check("nightly sweep runs", r["ok"], f"worst line {r['worst']}")

print("\nALL GOOD" if ok else "\nSOMETHING FAILED")
sys.exit(0 if ok else 1)
