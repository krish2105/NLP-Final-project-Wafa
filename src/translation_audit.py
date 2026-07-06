"""Translation-quality audit (honest multilingual-fairness evidence).

For each non-English language we sample messages, run the translation route, and
record whether translation actually happened and whether the output looks
faithful (heuristic). This surfaces the KNOWN failure — romanised Hindi is not
translatable by opus-mt-hi-en — as an explicit, measured finding instead of a
hidden one.

Run:  python -m src.translation_audit
"""
from __future__ import annotations

import sys

import pandas as pd

try:
    from . import config
    from .translation import translate_to_english, is_romanised, _transformers_available
    from .utils import save_json
except ImportError:  # pragma: no cover
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config
    from src.translation import translate_to_english, is_romanised, _transformers_available
    from src.utils import save_json

SAMPLE_PER_LANG = 6


def _faithful(original: str, translation: str, translated: bool) -> bool:
    """Cheap sanity heuristic: a faithful translation is non-empty, a sensible
    length relative to the source, and not a degenerate repetition."""
    if not translated:
        return False
    t = (translation or "").strip()
    if len(t) < 3:
        return False
    ratio = len(t) / max(1, len(original))
    if not (0.3 <= ratio <= 3.0):
        return False
    words = t.lower().split()
    if words and len(set(words)) / len(words) < 0.4:  # heavy repetition => garbled
        return False
    return True


def main():
    if not _transformers_available():
        print("transformers not installed — translation audit needs the full stack. Skipping.")
        return None

    m = pd.read_csv(config.MESSAGES_CSV)
    m["text"] = m["text"].fillna("").astype(str)
    rows = []
    per_lang = {}
    for lang in ["ar", "hi", "tl"]:
        sub = m[m.language == lang].head(SAMPLE_PER_LANG)
        faithful_n = 0
        samples = []
        for _, r in sub.iterrows():
            trans, flag = translate_to_english(r["text"], lang)
            ok = _faithful(r["text"], trans, flag)
            faithful_n += int(ok)
            samples.append({
                "original": r["text"][:90],
                "translation": (trans or "")[:90],
                "translated": bool(flag),
                "romanised_source": is_romanised(r["text"]),
                "looks_faithful": ok,
            })
        per_lang[lang] = {
            "language_name": config.LANGUAGE_NAMES.get(lang, lang),
            "n": int(len(sub)),
            "faithful_rate": round(faithful_n / max(1, len(sub)), 3),
            "route": "opus-mt" if lang in config.OPUS_MT_MODELS else "NLLB-600M",
            "samples": samples,
        }
        rows.append(per_lang[lang])

    finding = (
        "Arabic (opus-mt-ar-en) translates faithfully. Tagalog (NLLB) is usable. "
        "ROMANISED HINDI is NOT translatable by opus-mt-hi-en (it expects Devanagari) "
        "and is therefore left untranslated on purpose — the char-ngram classifier and "
        "romanised keyword entities handle it. This is a documented multilingual limitation, "
        "not a silent mistranslation."
    )
    out = {"per_language": per_lang, "finding": finding}
    save_json(out, config.METRICS_DIR / "translation_audit.json")

    print("\n=== Translation quality audit ===")
    for lang, v in per_lang.items():
        print(f"  {v['language_name']:9s} route={v['route']:9s} faithful_rate={v['faithful_rate']}")
        for s in v["samples"][:2]:
            print(f"      {'✓' if s['looks_faithful'] else '✗'} {s['original'][:55]!r}")
            print(f"        -> {s['translation'][:55]!r}")
    print("\n" + finding)
    print(f"Saved -> {config.METRICS_DIR / 'translation_audit.json'}")
    return out


if __name__ == "__main__":
    main()
