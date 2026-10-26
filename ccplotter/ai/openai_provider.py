import os
from pathlib import Path

from ccplotter.ai.comfy import _ai_cfg

# ── OpenAI line-art provider (cloud; replaces local ComfyUI when configured) ──
def _openai_key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    if k: return k.strip()
    try:
        f = Path("openai_key.txt")
        return f.read_text().strip() if f.exists() else ""
    except Exception: return ""

def _use_openai() -> bool:
    return _ai_cfg().get("provider", "openai") == "openai" and bool(_openai_key())

def _openai_lineart_bytes(pim, prompt: str, cfgd: dict) -> bytes:
    """One photo->line-art pass via OpenAI image edit. Returns PNG bytes — the
    same contract as the ComfyUI _gen, so all downstream cleanup/trace is reused.
    Output size matches the input aspect (closest gpt-image-1 bucket)."""
    import io, base64
    from openai import OpenAI
    w, h = pim.size; r = w / max(h, 1)
    size = "1536x1024" if r > 1.2 else ("1024x1536" if r < 0.83 else "1024x1024")
    buf = io.BytesIO(); pim.convert("RGB").save(buf, format="PNG"); buf.seek(0); buf.name = "input.png"
    client = OpenAI(api_key=_openai_key())
    res = client.images.edit(model=cfgd.get("openai_model", "gpt-image-1"), image=buf,
                             prompt=prompt, size=size, quality=cfgd.get("openai_quality", "medium"))
    return base64.b64decode(res.data[0].b64_json)
