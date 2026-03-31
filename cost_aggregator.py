#!/usr/bin/env python3
"""
Daily API cost aggregator — pulls usage/cost from each provider and writes to samaritan_costs.

Usage:
    python3 cost_aggregator.py              # aggregate today (or yesterday if run early)
    python3 cost_aggregator.py 2026-03-29   # specific date
    python3 cost_aggregator.py --backfill 7 # last 7 days

Providers implemented:
    - Google Gemini (Cloud Billing API — requires API enabled on project)
    - Tavily (usage endpoint)
    - Bright Data (balance endpoint)
    - Deepgram (usage endpoint)
    - xAI (accumulated from samaritan_tool_stats per-call cost logging)
"""

import json
import os
import sys
import datetime
import requests
import mysql.connector
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent / ".env"

def _load_env():
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)

_load_env()

# Also load webfe .env for DEEPGRAM_API_KEY (not in llmem-gw's .env)
_WEBFE_ENV = Path("/home/markj/projects/samaritan-webfe/.env")
if _WEBFE_ENV.exists():
    for line in _WEBFE_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _db_connect():
    return mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database="mymcp",
    )

def upsert_cost(date, provider, service, cost_usd, calls=0, tokens_in=0, tokens_out=0,
                unit=None, unit_count=None, raw_json=None):
    """Insert or update a cost row. Uses ON DUPLICATE KEY UPDATE."""
    conn = _db_connect()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO samaritan_costs
                (date, provider, service, calls, tokens_in, tokens_out, cost_usd, unit, unit_count, raw_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                calls = VALUES(calls),
                tokens_in = VALUES(tokens_in),
                tokens_out = VALUES(tokens_out),
                cost_usd = VALUES(cost_usd),
                unit = VALUES(unit),
                unit_count = VALUES(unit_count),
                raw_json = VALUES(raw_json)
        """, (date, provider, service, calls, tokens_in, tokens_out, cost_usd,
              unit, unit_count, json.dumps(raw_json) if raw_json else None))
        conn.commit()
        return True
    except Exception as e:
        print(f"  [ERROR] DB upsert failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()


# ── Provider: Tavily ───────────────────────────────────────────────────────────

def fetch_tavily(date):
    """Tavily usage API — returns search/extract/crawl counts."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("  [SKIP] TAVILY_API_KEY not set")
        return

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.get("https://api.tavily.com/usage", headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Response has "key" (per-key usage) and "account" (plan-level) objects
            key_data = data.get("key", data)
            acct_data = data.get("account", {})
            search_usage = key_data.get("search_usage", key_data.get("usage", 0))
            extract_usage = key_data.get("extract_usage", 0)
            total_calls = search_usage + extract_usage
            plan_limit = acct_data.get("plan_limit", 0)
            plan_usage = acct_data.get("plan_usage", 0)

            upsert_cost(
                date=date, provider="tavily", service="search-api",
                cost_usd=0,  # prepaid credits — cost tracked via balance
                calls=total_calls,
                unit="api_calls", unit_count=total_calls,
                raw_json=data,
            )
            remaining = plan_limit - plan_usage if plan_limit else "unlimited"
            print(f"  [OK] Tavily: {total_calls} calls (search={search_usage}, extract={extract_usage}), remaining={remaining}")
        else:
            print(f"  [WARN] Tavily usage endpoint returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] Tavily: {e}")


# ── Provider: Bright Data ─────────────────────────────────────────────────────

def fetch_brightdata(date):
    """Bright Data balance endpoint."""
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        print("  [SKIP] BRIGHTDATA_API_KEY not set")
        return

    try:
        resp = requests.get(
            "https://api.brightdata.com/customer/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            balance = data.get("balance", 0)
            pending = data.get("pending_costs", 0)
            upsert_cost(
                date=date, provider="brightdata", service="scraping-api",
                cost_usd=pending,
                unit="usd_pending", unit_count=pending,
                raw_json=data,
            )
            print(f"  [OK] Bright Data: balance=${balance:.2f}, pending=${pending:.2f}")
        else:
            print(f"  [WARN] Bright Data returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] Bright Data: {e}")


# ── Provider: Deepgram ─────────────────────────────────────────────────────────

def fetch_deepgram(date):
    """Deepgram usage API — requires project ID lookup first."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("  [SKIP] DEEPGRAM_API_KEY not set")
        return

    headers = {"Authorization": f"Token {api_key}"}
    try:
        # Get project ID
        resp = requests.get("https://api.deepgram.com/v1/projects", headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  [WARN] Deepgram projects returned {resp.status_code}: {resp.text[:200]}")
            return

        projects = resp.json().get("projects", [])
        if not projects:
            print("  [WARN] Deepgram: no projects found")
            return

        project_id = projects[0]["project_id"]

        # Get usage for the date (Deepgram accepts YYYY-MM-DD)
        end_dt = datetime.datetime.strptime(str(date), "%Y-%m-%d") + datetime.timedelta(days=1)

        resp = requests.get(
            f"https://api.deepgram.com/v1/projects/{project_id}/usage",
            headers=headers,
            params={"start": str(date), "end": end_dt.strftime("%Y-%m-%d")},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Extract totals from usage response
            results = data.get("results", [])
            total_hours = sum(r.get("hours", 0) for r in results)
            total_requests = sum(r.get("requests", 0) for r in results)
            total_tokens_in = sum(r.get("tokens", {}).get("in", 0) for r in results)
            total_tokens_out = sum(r.get("tokens", {}).get("out", 0) for r in results)

            # Deepgram pay-as-you-go: ~$0.0043/min for Nova-2
            estimated_cost = total_hours * 60 * 0.0043

            upsert_cost(
                date=date, provider="deepgram", service="stt-tts",
                cost_usd=estimated_cost,
                calls=total_requests,
                tokens_in=total_tokens_in, tokens_out=total_tokens_out,
                unit="audio_hours", unit_count=total_hours,
                raw_json=data,
            )
            print(f"  [OK] Deepgram: {total_requests} requests, {total_hours:.2f}h, ~${estimated_cost:.4f}")
        else:
            print(f"  [WARN] Deepgram usage returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] Deepgram: {e}")


# ── Provider: Google Cloud (Gemini via Cloud Monitoring API) ──────────────────

_GCP_PROJECT = "gen-lang-client-0630273538"
_GCP_PROJECT_NUM = "492544606644"

# Gemini pricing (per 1M tokens) — updated March 2026
# Source: ai.google.dev/gemini-api/docs/pricing
# NOTE: Cloud Monitoring API cannot distinguish models in "by_method" breakdown —
# all calls appear as "unknown". These estimates assume gemini-2.5-flash as primary.
# If Pro models are in use, actual costs will be significantly higher (verified:
# actual costs can be 3-10x estimates depending on model mix).
_GEMINI_PRICING = {
    "gemini-2.5-flash":      {"input": 0.50, "output": 3.00},   # updated from 0.15/0.60
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash":      {"input": 0.10, "output": 0.40},
    "gemini-3.1-pro-preview":{"input": 2.00, "output": 15.00},
}

def fetch_google_monitoring(date):
    """Pull Gemini API request counts from Cloud Monitoring API.
    Requires Cloud Monitoring API enabled on project.
    """
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path or not Path(sa_path).exists():
        print("  [SKIP] GOOGLE_SERVICE_ACCOUNT_JSON not set or missing")
        return

    try:
        from google.oauth2 import service_account
        from google.cloud import monitoring_v3

        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform",
                     "https://www.googleapis.com/auth/monitoring.read"],
        )

        client = monitoring_v3.MetricServiceClient(credentials=creds)

        # Build time interval for the target date
        date_obj = datetime.datetime.strptime(str(date), "%Y-%m-%d")
        start_ts = int(date_obj.replace(tzinfo=datetime.timezone.utc).timestamp())
        end_ts = start_ts + 86400

        interval = monitoring_v3.TimeInterval({
            "start_time": {"seconds": start_ts},
            "end_time": {"seconds": end_ts},
        })

        # Query request count for generativelanguage.googleapis.com
        results = client.list_time_series(
            request={
                "name": f"projects/{_GCP_PROJECT}",
                "filter": (
                    'metric.type = "serviceruntime.googleapis.com/api/request_count"'
                    ' AND resource.type = "consumed_api"'
                    ' AND resource.labels.service = "generativelanguage.googleapis.com"'
                ),
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )

        total_requests = 0
        by_method = {}
        for ts in results:
            method = ts.metric.labels.get("method", "unknown")
            count = sum(p.value.int64_value for p in ts.points)
            by_method[method] = by_method.get(method, 0) + count
            total_requests += count

        if total_requests == 0:
            print(f"  [INFO] Google Monitoring: no Gemini requests found for {date}")
            return

        # Get actual data volumes from request/response size metrics
        total_input_bytes = 0
        total_output_bytes = 0
        for metric_suffix, label in [("api/request_sizes", "input"), ("api/response_sizes", "output")]:
            size_results = client.list_time_series(request={
                "name": f"projects/{_GCP_PROJECT}",
                "filter": (
                    f'metric.type = "serviceruntime.googleapis.com/{metric_suffix}"'
                    ' AND resource.type = "consumed_api"'
                    ' AND resource.labels.service = "generativelanguage.googleapis.com"'
                ),
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            })
            for ts in size_results:
                for point in ts.points:
                    d = point.value.distribution_value
                    if label == "input":
                        total_input_bytes += d.mean * d.count
                    else:
                        total_output_bytes += d.mean * d.count

        # ~4 bytes per token for JSON/text payloads
        est_tokens_in = total_input_bytes / 4
        est_tokens_out = total_output_bytes / 4

        # Cost using gemini-2.5-flash pricing (dominant model)
        pricing = _GEMINI_PRICING["gemini-2.5-flash"]
        est_cost = (est_tokens_in * pricing["input"] + est_tokens_out * pricing["output"]) / 1_000_000

        upsert_cost(
            date=date, provider="google", service="gemini-api-monitoring",
            cost_usd=est_cost,
            calls=total_requests,
            tokens_in=int(est_tokens_in),
            tokens_out=int(est_tokens_out),
            unit="requests_from_monitoring", unit_count=total_requests,
            raw_json={
                "by_method": by_method,
                "input_bytes": int(total_input_bytes),
                "output_bytes": int(total_output_bytes),
                "est_tokens_in": int(est_tokens_in),
                "est_tokens_out": int(est_tokens_out),
                "pricing_model": "gemini-2.5-flash",
            },
        )
        avg_in_kb = (total_input_bytes / total_requests / 1024) if total_requests else 0
        print(f"  [OK] Google Monitoring: {total_requests} requests, avg {avg_in_kb:.1f}KB input")
        print(f"       Tokens: ~{est_tokens_in/1000:.0f}K in, ~{est_tokens_out/1000:.0f}K out")
        print(f"       Estimated cost: ${est_cost:.4f}  [WARNING: model mix unknown — check AI Studio for actual]")

    except ImportError:
        print("  [SKIP] google-cloud-monitoring not installed (pip install google-cloud-monitoring)")
    except Exception as e:
        err_str = str(e)
        if "SERVICE_DISABLED" in err_str:
            print(f"  [SKIP] Cloud Monitoring API not enabled on project {_GCP_PROJECT_NUM}")
            print(f"         Enable: https://console.developers.google.com/apis/api/monitoring.googleapis.com/overview?project={_GCP_PROJECT_NUM}")
            # Fall back to internal estimate
            _fetch_gemini_internal_estimate(date)
        else:
            print(f"  [ERROR] Google Monitoring: {e}")


def _fetch_gemini_internal_estimate(date):
    """Fallback: estimate Gemini costs from internal samaritan_tool_stats + LLM call counts."""
    conn = _db_connect()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT model, tool_name, call_count, success_count
            FROM samaritan_tool_stats
            WHERE model LIKE '%%gemini%%' OR model LIKE '%%summarizer%%'
                OR model LIKE '%%reviewer%%' OR model LIKE '%%extractor%%'
                OR model LIKE '%%judge%%'
        """)
        rows = cur.fetchall()
        if not rows:
            print("  [INFO] Gemini fallback: no internal call stats found")
            return

        total_calls = sum(r["call_count"] for r in rows)
        # Estimate: 1000 in + 400 out tokens per call at flash pricing
        # WARNING: model mix unknown — actual costs may be higher if Pro models are in use
        pricing = _GEMINI_PRICING["gemini-2.5-flash"]
        est_cost = total_calls * ((1000 * pricing["input"] + 400 * pricing["output"]) / 1_000_000)

        upsert_cost(
            date=date, provider="google", service="gemini-internal-estimate",
            cost_usd=est_cost,
            calls=total_calls,
            unit="estimated_from_tool_stats", unit_count=total_calls,
            raw_json={"models": [dict(r) for r in rows]},
        )
        print(f"  [OK] Gemini fallback estimate: {total_calls} tracked calls, ~${est_cost:.4f}")
        print(f"       NOTE: internal roles (memreview, cognition) may not be tracked here")
    except Exception as e:
        print(f"  [ERROR] Gemini fallback: {e}")
    finally:
        cur.close()
        conn.close()


# ── Provider: xAI (from per-call cost_in_usd_ticks in responses) ──────────────

def fetch_xai_accumulated(date):
    """xAI costs from accumulated per-call logging in tool stats.
    xAI responses include cost_in_usd_ticks — accumulated via llm_call handling.
    """
    conn = _db_connect()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT model, call_count, success_count
            FROM samaritan_tool_stats
            WHERE model LIKE '%%xai%%' OR model LIKE '%%grok%%'
        """)
        rows = cur.fetchall()
        if rows:
            total_calls = sum(r["call_count"] for r in rows)
            # xAI Grok pricing varies; log call count for now
            upsert_cost(
                date=date, provider="xai", service="grok-api",
                cost_usd=0,  # actual cost needs per-call accumulation
                calls=total_calls,
                unit="api_calls", unit_count=total_calls,
                raw_json={"models": [dict(r) for r in rows]},
            )
            print(f"  [OK] xAI: {total_calls} total calls (actual cost needs per-call logging)")
        else:
            print("  [INFO] xAI: no call stats found")
    except Exception as e:
        print(f"  [ERROR] xAI: {e}")
    finally:
        cur.close()
        conn.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def fetch_per_call_events(date):
    """Pull per-call cost events from samaritan_cost_events and upsert into samaritan_costs.
    This captures costs logged in real-time by tool wrappers (search_tavily, search_brightdata,
    search_xai, llm_call for Gemini/xAI models).
    """
    try:
        from cost_events import fetch_events_for_date
        rows = fetch_events_for_date(date)
        if not rows:
            print(f"  [INFO] Per-call events: no events found for {date}")
            return
        for row in rows:
            upsert_cost(
                date=date,
                provider=row["provider"],
                service=f"{row['service']}-events",
                cost_usd=float(row["cost_usd"]),
                calls=int(row["calls"]),
                tokens_in=int(row["tokens_in"]),
                tokens_out=int(row["tokens_out"]),
                unit="per_call_events",
                unit_count=int(row["calls"]),
                raw_json={"source": "samaritan_cost_events", "date": str(date)},
            )
            print(f"  [OK] Per-call events: {row['provider']}/{row['service']} — "
                  f"{row['calls']} calls, ~${float(row['cost_usd']):.4f}")
    except Exception as e:
        print(f"  [ERROR] Per-call events: {e}")


ALL_PROVIDERS = [
    ("Per-call Events (real-time)", fetch_per_call_events),
    ("Tavily", fetch_tavily),
    ("Bright Data", fetch_brightdata),
    ("Deepgram", fetch_deepgram),
    ("Google Gemini (Monitoring)", fetch_google_monitoring),
    ("xAI", fetch_xai_accumulated),
]


def run_for_date(date):
    print(f"\n{'='*60}")
    print(f"Cost aggregation for {date}")
    print(f"{'='*60}")
    for name, func in ALL_PROVIDERS:
        print(f"\n[{name}]")
        func(date)
    print(f"\n{'='*60}")
    print(f"Done for {date}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        today = datetime.date.today()
        for i in range(days, 0, -1):
            run_for_date(today - datetime.timedelta(days=i))
    elif len(sys.argv) > 1:
        run_for_date(sys.argv[1])
    else:
        run_for_date(datetime.date.today())


if __name__ == "__main__":
    main()
