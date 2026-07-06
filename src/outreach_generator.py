"""Outreach draft generation with a graceful fallback chain.

Fallback order (log which one actually loaded):
  1. Qwen/Qwen2.5-1.5B-Instruct   (4-bit if bitsandbytes available)
  2. Qwen/Qwen2.5-0.5B-Instruct
  3. google/flan-t5-base
  4. Rule-based templates          (always works — no model needed)

The platform must ALWAYS produce a coherent draft, even fully offline, so the
rule-based generator is a first-class citizen, not an afterthought.

Every draft is returned with `human_status = "Pending Review"` and is run
through guardrails before being surfaced. No auto-send path exists.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from . import config
from .guardrails import check_draft

logger = logging.getLogger("wafa.outreach")

# --------------------------------------------------------------------------- #
# The system prompt is a named constant and is exported verbatim into the
# Ethics Statement (Section 7 requires "show us the prompt").
# --------------------------------------------------------------------------- #
OUTREACH_SYSTEM_PROMPT = """You are a retention specialist at a UAE bank drafting a short outreach message.
Rules you must follow, without exception:
- Never use urgency language ("limited time", "act now", "expires soon").
- Never say the customer "must" stay or imply any obligation to remain a customer.
- Never state a fee, rate, or offer value beyond what is explicitly given to you.
- Never reference the customer's anxiety, distress, or emotional state to persuade them.
- Never invent a product name that was not given to you.
- If the action is "Dignified Goodbye Pathway": do NOT offer any retention incentive.
  Thank the customer, offer help with a smooth transfer/closure, and preserve goodwill only.
- Be honest, warm, professional, and concise (3-5 sentences).
- Match the customer's language when a translation model is available; otherwise write in English
  and note the target language for human translation.
"""

# --------------------------------------------------------------------------- #
# Rule-based templates keyed by action. Placeholders are filled from the
# nlp_output + decision so the draft is always concrete and on-policy.
# --------------------------------------------------------------------------- #
_TEMPLATES = {
    "Dignified Goodbye Pathway": (
        "Dear valued customer, thank you sincerely for banking with Falcon Bank. "
        "We understand you may be moving on, and we want your transition to be as smooth as possible. "
        "If it helps, we can assist with closing your account, transferring your balance, or providing "
        "any final statements you need. It has been our privilege to serve you, and you are always welcome back."
    ),
    "Relationship Manager Call": (
        "Dear valued customer, thank you for being with Falcon Bank. "
        "One of our relationship managers would like to speak with you personally to understand how we can "
        "better support your banking needs. Please let us know a convenient time and we will arrange a call."
    ),
    "Fee Waiver / Service Recovery": (
        "Dear valued customer, thank you for raising your concern about {issue_human}. "
        "We are sorry for the inconvenience and would like to make this right. "
        "A member of our service team will review your account and follow up with a resolution. "
        "We appreciate your patience and your continued trust in Falcon Bank."
    ),
    "Remittance Support": (
        "Dear valued customer, thank you for reaching out about your international transfer. "
        "Our remittance service team can guide you through the process and help ensure your funds reach their "
        "destination smoothly. Please let us know if you would like a callback or assistance in branch."
    ),
    "Loan Settlement Advisory": (
        "Dear valued customer, thank you for your enquiry regarding your loan/mortgage. "
        "Our advisory team can walk you through your options, including settlement steps if you are relocating. "
        "We are happy to arrange a convenient time to discuss the details with you."
    ),
    "Service Fix": (
        "Dear valued customer, thank you for letting us know about the issue you experienced with your {issue_human}. "
        "Our team is looking into it and will help resolve the friction as quickly as possible. "
        "Please reach out if there is anything further we can do to assist."
    ),
    "Standard Information Reply": (
        "Dear valued customer, thank you for your message. "
        "We are happy to help with your query and have shared the information you requested. "
        "Please don't hesitate to contact us again if you need anything else from Falcon Bank."
    ),
    "Retention Offer with Economics Check": (
        "Dear valued customer, thank you for banking with Falcon Bank. "
        "We value your relationship with us and would welcome the chance to review your account together "
        "to make sure it continues to serve you well. A member of our team will be in touch to discuss how we can help."
    ),
}

_ISSUE_HUMAN = {
    "Account_Closure": "account closure",
    "Remittance_Transfer": "international transfer",
    "Loan_Mortgage": "loan or mortgage",
    "Fees_Charges": "the fees on your account",
    "Card_Services": "your card",
    "App_Technical": "the mobile app",
    "General_Query": "your query",
}


# --------------------------------------------------------------------------- #
# Model loading (lazy, cached). Any failure quietly degrades to the next tier.
# --------------------------------------------------------------------------- #
class _ModelState:
    loaded_name: Optional[str] = None
    kind: Optional[str] = None  # "causal" | "seq2seq" | "template"
    tokenizer = None
    model = None
    tried = False


_state = _ModelState()


def get_active_generator_name() -> str:
    """Which generator is (or would be) used, without forcing a full load."""
    if _state.tried:
        return _state.loaded_name or "rule-based-template"
    return "not yet loaded"


def _try_load_models() -> None:
    if _state.tried:
        return
    _state.tried = True

    if config.WAFA_LIGHT_MODE:
        logger.info("WAFA_LIGHT_MODE=1 -> using rule-based templates only.")
        _state.loaded_name = "rule-based-template"
        _state.kind = "template"
        return

    try:
        import torch  # noqa: F401
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
    except Exception as e:  # transformers/torch not available
        logger.warning("transformers unavailable (%s) -> rule-based templates.", e)
        _state.loaded_name = "rule-based-template"
        _state.kind = "template"
        return

    for name in config.OUTREACH_MODEL_CHAIN:
        try:
            logger.info("Attempting to load outreach model: %s", name)
            tok = AutoTokenizer.from_pretrained(name)
            if "flan-t5" in name.lower():
                mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
                _state.kind = "seq2seq"
            else:
                mdl = AutoModelForCausalLM.from_pretrained(name)
                _state.kind = "causal"
            _state.tokenizer = tok
            _state.model = mdl
            _state.loaded_name = name
            logger.info("Loaded outreach model: %s", name)
            return
        except Exception as e:
            logger.warning("Could not load %s (%s) -> trying next.", name, e)

    logger.warning("No outreach LLM loaded -> rule-based templates.")
    _state.loaded_name = "rule-based-template"
    _state.kind = "template"


def _build_user_prompt(nlp_output: dict, decision: dict, customer_row: dict,
                       target_language: str = "en") -> str:
    issue_human = _ISSUE_HUMAN.get(nlp_output.get("issue_type", ""), "your query")
    offer_line = (
        f"You may reference a goodwill gesture up to AED {decision['max_offer_value_aed']:g} "
        "(do not state a specific number unless necessary)."
        if decision.get("offer_allowed")
        else "Do NOT reference any monetary offer or incentive."
    )
    goodbye_line = (
        "This customer is leaving — write a warm, pressure-free farewell: thank them, "
        "offer help with a smooth account closure/transfer, and wish them well. Offer NO incentive."
        if decision.get("dignified_goodbye")
        else "Acknowledge their topic and offer a concrete, helpful next step."
    )
    lang_line = ""
    if target_language and target_language != "en":
        lang_line = f"Write the ENTIRE message in {config.LANGUAGE_NAMES.get(target_language, target_language)}.\n"
    return (
        f"Draft a short customer outreach message for a bank.\n"
        f"Internal action (context only — do NOT name it in the message): {decision['action']}\n"
        f"Customer topic: {issue_human}\n"
        f"Customer segment: {customer_row.get('segment', 'Mass')}\n"
        f"{offer_line}\n"
        f"{goodbye_line}\n"
        f"{lang_line}"
        f"Rules: do not mention internal system/pathway/action names; do not invent "
        f"products, fees, or amounts; write 3-5 sentences, warm and professional."
    )


def _template_draft(nlp_output: dict, decision: dict) -> str:
    tmpl = _TEMPLATES.get(decision["action"], _TEMPLATES["Standard Information Reply"])
    issue_human = _ISSUE_HUMAN.get(nlp_output.get("issue_type", ""), "your query")
    return tmpl.format(issue_human=issue_human)


def _llm_draft(nlp_output: dict, decision: dict, customer_row: dict,
               target_language: str = "en") -> Optional[str]:
    try:
        import torch

        user = _build_user_prompt(nlp_output, decision, customer_row, target_language)
        if _state.kind == "causal":
            messages = [
                {"role": "system", "content": OUTREACH_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            prompt = _state.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = _state.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = _state.model.generate(
                    **inputs, max_new_tokens=180, do_sample=False, temperature=None, top_p=None
                )
            gen = out[0][inputs["input_ids"].shape[1] :]
            return _state.tokenizer.decode(gen, skip_special_tokens=True).strip()
        else:  # seq2seq (flan-t5)
            prompt = OUTREACH_SYSTEM_PROMPT + "\n\n" + user
            inputs = _state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = _state.model.generate(**inputs, max_new_tokens=180)
            return _state.tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.warning("LLM generation failed (%s) -> template fallback.", e)
        return None


def generate_outreach(
    nlp_output: dict, decision: dict, customer_row: dict, use_llm: bool = True,
    match_language: bool = False,
) -> Dict:
    """Produce an OutreachOutput dict (always human-review-gated).

    match_language=True asks the LLM (Qwen, if loaded) to draft in the customer's
    own language — a stretch feature (brief: "outreach in the customer's language").
    Only supported by the causal LLM tier; templates/Flan stay English.
    """
    draft_text: Optional[str] = None
    customer_lang = nlp_output.get("language", "en")
    drafted_language = "en"

    if use_llm:
        _try_load_models()
        if _state.kind in ("causal", "seq2seq"):
            want_lang = customer_lang if (match_language and _state.kind == "causal") else "en"
            draft_text = _llm_draft(nlp_output, decision, customer_row, target_language=want_lang)
            if draft_text and want_lang != "en":
                drafted_language = want_lang

    if not draft_text:
        draft_text = _template_draft(nlp_output, decision)

    guard = check_draft(draft_text, decision)

    language_note = ""
    if drafted_language == "en" and customer_lang != "en":
        language_note = (
            f"\n\n[Draft written in English — flag for human translation to "
            f"{config.LANGUAGE_NAMES.get(customer_lang, customer_lang)}.]"
        )
    elif drafted_language != "en":
        guard["guardrail_warnings"].append(
            f"Draft is in {config.LANGUAGE_NAMES.get(drafted_language, drafted_language)} — "
            "keyword guardrails are English-oriented, so a human fluent in the language must review."
        )

    return {
        "draft_language": drafted_language,
        "draft_text": draft_text + language_note,
        "guardrail_passed": guard["guardrail_passed"],
        "guardrail_warnings": guard["guardrail_warnings"],
        "human_status": "Pending Review",
        "generator_used": get_active_generator_name(),
    }
