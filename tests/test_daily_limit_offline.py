"""Offline test: a daily (TPD) 429 fails over to the next vision model, and build_index keeps its progress.
Run:  python tests/test_daily_limit_offline.py"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np

from adagent import assets, llm

os.environ["GROQ_API_KEY"] = "test"
llm.time.sleep = lambda s: None
llm._llm_cfg.cache_clear()
llm._llm_cfg = lambda: {"provider": "groq", "vision_fallbacks": ["fallback/vision-b"]}

TPD = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization on tokens per day (TPD): "
       "Limit 200000, Used 197621, Requested 2465. Please try again in 38s.")


class R:
    def __init__(self, code, js=None, text="", headers=None):
        self.status_code, self._js, self.text, self.headers = code, js, text or str(js), headers or {}

    def json(self):
        return self._js


def ok(txt):
    return R(200, {"choices": [{"message": {"content": txt}}]})


sent, seq = [], []


def fake_post(url, headers=None, json=None, timeout=0):
    sent.append(copy.deepcopy(json))
    return seq.pop(0)


llm.requests.post = fake_post
img = np.zeros((64, 64, 3), np.uint8)

# 1. TPD 429 on the main model -> NO waiting/retrying, straight to the fallback model
seq[:] = [R(429, text=TPD), ok('{"description":"ok"}')]
out = llm.ask("main/vision-a", "sys", "usr", images=[img])
assert out == '{"description":"ok"}'
assert [b["model"] for b in sent] == ["main/vision-a", "fallback/vision-b"], [b["model"] for b in sent]
assert "main/vision-a" in llm._exhausted

# 2. an exhausted model is skipped on later calls (no wasted request)
sent.clear(); seq[:] = [ok('{"description":"again"}')]
llm.ask("main/vision-a", "sys", "usr", images=[img])
assert [b["model"] for b in sent] == ["fallback/vision-b"]

# 3. when every model is exhausted -> DailyLimitError
sent.clear(); seq[:] = [R(429, text=TPD)]
try:
    llm.ask("main/vision-a", "sys", "usr", images=[img])
    raise AssertionError("should have raised")
except llm.DailyLimitError:
    pass

# 3b. "Request too large ... OTPM Limit 1000, Requested 1316": lower max_tokens at once (no waiting), then succeed
OTPM = ("Request too large for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on output "
        "tokens per minute (OTPM): Limit 1000, Requested 1316. The request's expected output tokens exceed the "
        "enforced limit; reduce max_tokens")
sent.clear(); seq[:] = [R(429, text=OTPM), ok('{"description":"fits"}')]
slept = []
llm.time.sleep = lambda s: slept.append(s)
out = llm._ask_groq("size/model", "sys", "usr", [img], 1500, 0.0)
assert out == '{"description":"fits"}' and not slept, slept
assert len(sent) == 2 and sent[1]["max_tokens"] < 1000 and sent[1]["max_tokens"] < sent[0]["max_tokens"], [b["max_tokens"] for b in sent]
# an INPUT-side "request too large" cannot be fixed by max_tokens: fail fast with a clear message
sent.clear(); seq[:] = [R(429, text="Request too large for model x on tokens per minute (TPM): Limit 6000, Requested 9000.")]
try:
    llm._ask_groq("size/model2", "sys", "usr", [img], 300, 0.0)
    raise AssertionError("should have raised")
except RuntimeError as e:
    assert "waiting will not help" in str(e) and len(sent) == 1

# 3c. per-minute TPM 429 repeated 10 times: waits and succeeds instead of giving up after 8 tries; spaces later calls
TPM = ("Rate limit reached for model `qwen/qwen3.8-27b` on tokens per minute (TPM): Limit 8000, Used 6873, "
       "Requested 2596. Please try again in 11.0175s.")
llm._tpm_gap.clear(); llm._last_call.clear()
sent.clear(); seq[:] = [R(429, text=TPM) for _ in range(10)] + [ok('{"description":"finally"}')]
slept = []
llm.time.sleep = lambda s: slept.append(s)
out = llm._ask_groq("tpm/model", "sys", "usr", [img], 350, 0.0)
assert out == '{"description":"finally"}' and len(sent) == 11, len(sent)
waits = [w for w in slept if w < 15]            # the 429 waits (the others are the spacing between requests)
assert len(waits) == 10 and all(11 < w <= 13 for w in waits), slept
assert 20 < llm._tpm_gap["tpm/model"] < 23, llm._tpm_gap          # 2596/8000*60*1.1
llm.time.sleep = lambda s: None

# 4. build_index: progress is saved after each asset, then it stops cleanly; re-run only does what is left
d = Path(tempfile.mkdtemp())
(d / "assets").mkdir()
for n in ("a.png", "b.png", "c.png"):
    cv2.imwrite(str(d / "assets" / n), np.full((80, 80, 3), 40 + 60 * "abc".index(n[0]), np.uint8))


class P:  # minimal Project stand-in
    assets_dir, cache_dir, facts = d / "assets", d, {}


calls = []


def fake_describe(cfg, frames, kind):
    calls.append(1)
    if len(calls) == 3:
        raise llm.DailyLimitError("daily quota used up")
    return {"description": "d", "shows_product": True, "has_human_face": False,
            "has_price_or_rating_ui": False, "has_third_party_logo_or_watermark": False}


assets.describe = fake_describe
assets.haar_has_face = lambda frames: False
try:
    assets.build_index(P, {}, vision=True)
    raise AssertionError("should have stopped")
except SystemExit as e:
    assert "STOPPED" in str(e)
saved = json.loads((d / "assets_index.json").read_text())
assert len(saved) == 2, len(saved)               # a and b kept, c not yet described

calls.clear()
assets.describe = lambda cfg, frames, kind: (calls.append(1) or {"description": "d", "shows_product": True,
                                              "has_human_face": False, "has_price_or_rating_ui": False,
                                              "has_third_party_logo_or_watermark": False})
res = assets.build_index(P, {}, vision=True)
assert len(res) == 3 and len(calls) == 1, (len(res), len(calls))   # only the missing asset was described
# 5. a damaged video (e.g. "moov atom not found") is skipped with a clear reason; the run continues
d2 = Path(tempfile.mkdtemp()); (d2 / "assets").mkdir()
cv2.imwrite(str(d2 / "assets" / "good.png"), np.full((80, 80, 3), 120, np.uint8))
(d2 / "assets" / "2026-10-23 17-23-29 - Trim3.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)   # truncated: no moov atom


class P2:
    assets_dir, cache_dir, facts = d2 / "assets", d2, {}


res = assets.build_index(P2, {}, vision=False)
bad = [e for e in res if e["file"].endswith("Trim3.mp4")][0]
good = [e for e in res if e["file"] == "good.png"][0]
assert bad["usable"] is False and "damaged" in bad["reason"], bad
assert good["usable"] is True
# 6. affiliate link rule: official Amazon short links pass, look-alikes and third-party shorteners do not
from adagent.project import AMAZON_LINK_RE
for ok_url in ("https://link.amazon/B0esjLjpT", "https://amzn.to/abc", "https://a.co/d/x", "https://www.amazon.com/dp/B08S3M2HKL?tag=x-20"):
    assert AMAZON_LINK_RE.match(ok_url), ok_url
for bad_url in ("https://bit.ly/abc", "https://link.amazon.evil.com/x", "https://notlink.amazon/x", "https://evilamazon.com/x", "http://link.amazon/x"):
    assert not AMAZON_LINK_RE.match(bad_url), bad_url
print("daily-limit + incremental-cache tests passed")
