"""ccplotter/routes/bench.py -- Source Art Studio batch benchmark/evaluation
Flask routes: /api/studio/benchmark/start,
/api/studio/benchmark/status/<bid>, /api/studio/benchmark/cancel/<bid>,
/api/studio/benchmark/list, /api/studio/benchmark/<bid>,
/api/studio/benchmark/<bid>/rate, /api/studio/benchmark/<bid>/export.<fmt>,
/api/studio/benchmark/<bid> (DELETE), /api/studio/benchmark/<bid>/metrics,
/api/studio/benchmark/<bid>/critic_prompt/<int:idx>,
/api/studio/benchmark/<bid>/critic.

Shared state: _BENCH_JOBS is a dict mutated only by item assignment
(`_BENCH_JOBS[session["id"]] = {...}` in api_bench_start) and never rebound, so
importing it by name is safe. _AI_LOCK is shared with ai_clean.py/studio.py and
is likewise never rebound.
"""
import json
import os
import time
import threading
from uuid import uuid4

from flask import jsonify, request

from ccplotter.atomic_io import atomic_write_text
from ccplotter.config import STUDIO_PAIRS
from ccplotter.ai.clean import AI_CLEAN_SPEEDS, _AI_LOCK
from ccplotter.studio import STUDIO_CANDIDATE_STYLES, _studio_slug
from ccplotter.bench import (BENCH_DIR, BENCH_INDEX, BENCH_WIN_CATEGORIES, BENCH_RUBRIC,
                              BENCH_FAILURE_TAGS, _BENCH_JOBS, _bench_index_read,
                              _bench_load, _bench_save, _bench_metrics_for_pair,
                              BENCH_CRITIC_WIN_KEYS, BENCH_CRITIC_SUB, _bench_critic_prompt,
                              _bench_aggregate, _bench_worker, _bench_export_rows,
                              _bench_export_md)

from ccplotter.routes import app

@app.route("/api/studio/benchmark/start", methods=["POST"])
def api_bench_start():
    body = request.json or {}
    styles = [s for s in body.get("styles", list(STUDIO_CANDIDATE_STYLES)) if s in STUDIO_CANDIDATE_STYLES]
    if not styles:
        return jsonify({"error": "no valid styles"}), 400
    speed = body.get("speed", "fast")
    if speed not in AI_CLEAN_SPEEDS:
        return jsonify({"error": f"unknown speed '{speed}'"}), 400
    resume_id = body.get("resume_id")
    if resume_id:
        session = _bench_load(resume_id)
        if not session:
            return jsonify({"error": "benchmark session not found"}), 404
        session["resume"] = True
    else:
        try:
            pairs = json.loads(STUDIO_PAIRS.read_text())
        except Exception:
            pairs = []
        idxs = body.get("pair_indices")
        if idxs:
            pairs = [pairs[i] for i in idxs if 0 <= i < len(pairs)]
        if not pairs:
            return jsonify({"error": "no saved pairs to benchmark"}), 400
        bid = uuid4().hex[:8]
        session = {
            "id": bid, "label": (body.get("label") or time.strftime("%Y-%m-%d %H:%M")),
            "created": time.time(), "status": "running", "speed": speed,
            "crisp": bool(body.get("crisp", True)), "recover": bool(body.get("recover", False)),
            "styles": styles, "style_labels": {st: STUDIO_CANDIDATE_STYLES[st]["label"] for st in styles},
            "seed": int(body.get("seed", 0)) or int.from_bytes(os.urandom(4), "little"), "resume": False,
            "pairs": [{"index": i, "original": p.get("original"), "target": p.get("target"),
                       "note": p.get("note", ""), "case": _studio_slug(p.get("original", "")),
                       "candidates": [], "error": None, "rating": {}} for i, p in enumerate(pairs)],
        }
    if not _AI_LOCK.acquire(blocking=False):
        return jsonify({"error": "An AI job is already running — wait or cancel it."}), 429
    session["status"] = "running"
    _bench_save(session)
    _BENCH_JOBS[session["id"]] = {"cancel": False, "live": True, "current_pair": 0,
                                  "done_pairs": 0, "total": len(session["pairs"]),
                                  "phase": "queued", "pair_phase": "", "t0": time.time()}
    threading.Thread(target=_bench_worker, args=(session,), daemon=True).start()
    return jsonify({"bench_id": session["id"], "pairs": len(session["pairs"]),
                    "resumed": bool(resume_id)})

@app.route("/api/studio/benchmark/status/<bid>")
def api_bench_status(bid):
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    job = _BENCH_JOBS.get(bid, {})
    return jsonify({"bench_id": bid, "status": session.get("status"),
                    "phase": job.get("pair_phase") or job.get("phase", ""),
                    "current_pair": job.get("current_pair", 0),
                    "done_pairs": job.get("done_pairs", 0),
                    "total": len(session.get("pairs", [])),
                    "elapsed_s": round(time.time() - job["t0"], 1) if job.get("t0") else 0,
                    "error": session.get("error"), "live": job.get("live", False)})

@app.route("/api/studio/benchmark/cancel/<bid>", methods=["POST"])
def api_bench_cancel(bid):
    job = _BENCH_JOBS.get(bid)
    if job:
        job["cancel"] = True
    return jsonify({"ok": True})

@app.route("/api/studio/benchmark/list")
def api_bench_list():
    return jsonify({"sessions": _bench_index_read()})

@app.route("/api/studio/benchmark/<bid>")
def api_bench_get(bid):
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    session = dict(session)
    session["aggregate"] = _bench_aggregate(session)
    session["vocab"] = {"win_categories": BENCH_WIN_CATEGORIES, "rubric": BENCH_RUBRIC,
                        "failure_tags": BENCH_FAILURE_TAGS,
                        "critic_win_keys": BENCH_CRITIC_WIN_KEYS, "critic_sub": BENCH_CRITIC_SUB}
    session["live"] = _BENCH_JOBS.get(bid, {}).get("live", False)
    return jsonify(session)

@app.route("/api/studio/benchmark/<bid>/rate", methods=["POST"])
def api_bench_rate(bid):
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    body = request.json or {}
    pr = next((p for p in session["pairs"] if p["index"] == body.get("pair_index")), None)
    if pr is None:
        return jsonify({"error": "pair not found"}), 404
    pr["rating"] = {"ratings": body.get("ratings", {}), "rubric": body.get("rubric", {}),
                    "failures": body.get("failures", []), "note": body.get("note", ""), "t": time.time()}
    _bench_save(session)
    return jsonify({"ok": True, "aggregate": _bench_aggregate(session)})


@app.route("/api/studio/benchmark/<bid>/export.<fmt>")
def api_bench_export(bid, fmt):
    import csv, io, re
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", session.get("label", "") or bid)
    base = f"benchmark_{safe}_{bid}"
    if fmt == "json":
        out = dict(session); out["aggregate"] = _bench_aggregate(session)
        return json.dumps(out, indent=1), 200, {"Content-Type": "application/json",
               "Content-Disposition": f'attachment; filename="{base}.json"'}
    if fmt == "csv":
        rows = _bench_export_rows(session)
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        return buf.getvalue(), 200, {"Content-Type": "text/csv",
               "Content-Disposition": f'attachment; filename="{base}.csv"'}
    if fmt in ("md", "markdown"):
        return _bench_export_md(session), 200, {"Content-Type": "text/markdown; charset=utf-8",
               "Content-Disposition": f'attachment; filename="{base}.md"'}
    return jsonify({"error": f"unknown format '{fmt}'"}), 400

@app.route("/api/studio/benchmark/<bid>", methods=["DELETE"])
def api_bench_delete(bid):
    import shutil
    d = BENCH_DIR / bid
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    atomic_write_text(BENCH_INDEX, json.dumps([m for m in _bench_index_read() if m.get("id") != bid], indent=1))
    _BENCH_JOBS.pop(bid, None)
    return jsonify({"ok": True})

@app.route("/api/studio/benchmark/<bid>/metrics", methods=["POST"])
def api_bench_metrics(bid):
    """(Re)compute deterministic metrics for every pair — used to grade an older
    session that predates metrics, or to re-grade after editing targets."""
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    n = 0
    for pr in session["pairs"]:
        if pr.get("candidates"):
            try:
                m = _bench_metrics_for_pair(pr, session)
                if m:
                    pr["metrics"] = m; n += 1
            except Exception as e:
                print(f"  [bench] metrics recompute failed (pair {pr.get('index')}): {e}")
    _bench_save(session)
    return jsonify({"ok": True, "computed": n, "aggregate": _bench_aggregate(session)})

@app.route("/api/studio/benchmark/<bid>/critic_prompt/<int:idx>")
def api_bench_critic_prompt(bid, idx):
    """Return a ready-to-paste critic prompt + the image URLs for one pair, so
    the user can run it through a vision model and paste the JSON back."""
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    pr = next((p for p in session["pairs"] if p["index"] == idx), None)
    if pr is None or not pr.get("candidates"):
        return jsonify({"error": "pair not found or has no candidates"}), 404
    return jsonify({"prompt": _bench_critic_prompt(session, pr),
                    "original_url": f"/model_images/{pr['original']}" if pr.get("original") else None,
                    "target_url": f"/model_images/{pr['target']}" if pr.get("target") else None,
                    "candidates": [{"label": c["label"], "url": c["url"]} for c in pr["candidates"]]})

@app.route("/api/studio/benchmark/<bid>/critic", methods=["POST"])
def api_bench_critic(bid):
    """Save AI-critic JSON for one pair. Accepts either a parsed object or a raw
    JSON string (manual paste/import). A live API call can drop in here later."""
    session = _bench_load(bid)
    if not session:
        return jsonify({"error": "unknown benchmark"}), 404
    body = request.json or {}
    pr = next((p for p in session["pairs"] if p["index"] == body.get("pair_index")), None)
    if pr is None:
        return jsonify({"error": "pair not found"}), 404
    critic = body.get("critic")
    if isinstance(critic, str):
        try:
            critic = json.loads(critic)
        except Exception as e:
            return jsonify({"error": f"invalid critic JSON: {e}"}), 400
    if not isinstance(critic, dict):
        return jsonify({"error": "critic must be a JSON object"}), 400
    pr["critic"] = critic
    _bench_save(session)
    return jsonify({"ok": True, "aggregate": _bench_aggregate(session)})
