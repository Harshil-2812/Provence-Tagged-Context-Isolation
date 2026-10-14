"""
Response parser module for agent outputs.
Robustly parses LLM outputs, separating:
- Internal reasoning (<think> tags, deliberation blocks)
- Agent injection detections (INJECTION_DETECTED: prefix)
- Explicit and implicit final answers (ANSWER: prefix or direct text)
- Response status (ok, truncated, missing_answer, malformed)
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    raw_text: str
    clean_answer: Optional[str]
    reasoning: Optional[str]
    has_injection_detected: bool
    injection_detected_text: Optional[str]
    status: str  # "ok", "truncated", "missing_answer", "malformed"
    delimiter_count: int
    is_hypothetical_only: bool


def parse_llm_response(
    raw_text: str,
    max_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None
) -> ParsedResponse:
    """
    Parse a raw LLM output into reasoning, detection, and final answer components.
    
    Handles:
    - Qwen3 <think>...</think> reasoning blocks (closed and unclosed)
    - Single and multiple ANSWER: delimiters
    - Missing ANSWER: delimiters (extracts direct answer if complete, or marks missing)
    - Truncated deliberation outputs
    - Hypothetical option enumerations without an affirmative answer
    """
    if raw_text is None or not isinstance(raw_text, str) or not raw_text.strip():
        return ParsedResponse(
            raw_text=raw_text or "",
            clean_answer=None,
            reasoning=None,
            has_injection_detected=False,
            injection_detected_text=None,
            status="malformed",
            delimiter_count=0,
            is_hypothetical_only=False,
        )

    text = raw_text.strip()
    reasoning_parts = []
    
    # 1. Handle <think>...</think> blocks
    if "<think>" in text:
        if "</think>" in text:
            # Extract content between <think> and </think>
            parts = text.split("</think>", 1)
            think_content = parts[0].replace("<think>", "").strip()
            if think_content:
                reasoning_parts.append(think_content)
            remainder = parts[1].strip()
        else:
            # Unclosed <think> tag — model truncated while reasoning
            think_content = text.replace("<think>", "").strip()
            return ParsedResponse(
                raw_text=raw_text,
                clean_answer=None,
                reasoning=think_content,
                has_injection_detected=False,
                injection_detected_text=None,
                status="truncated",
                delimiter_count=0,
                is_hypothetical_only=True,
            )
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning_parts.append(parts[0].strip())
        remainder = parts[1].strip()
    else:
        remainder = text

    # 2. Check for INJECTION_DETECTED:
    has_injection_detected = "INJECTION_DETECTED:" in remainder or "INJECTION_DETECTED:" in text
    injection_text = None
    if "INJECTION_DETECTED:" in remainder:
        inj_parts = remainder.split("INJECTION_DETECTED:", 1)
        if inj_parts[0].strip():
            reasoning_parts.append(inj_parts[0].strip())
        after_inj = inj_parts[1].strip()
        if "ANSWER:" in after_inj:
            inj_desc, remainder = after_inj.split("ANSWER:", 1)
            injection_text = inj_desc.strip()
            remainder = "ANSWER:" + remainder
        else:
            injection_text = after_inj
            remainder = ""

    # 3. Handle ANSWER: delimiters
    answer_matches = list(re.finditer(r'(?i)\bANSWER:\s*', remainder))
    delimiter_count = len(answer_matches)
    
    clean_answer: Optional[str] = None
    status = "ok"
    is_hypothetical_only = False
    is_token_cap = (
        completion_tokens is not None and max_tokens is not None
        and completion_tokens >= max_tokens
    )

    if delimiter_count > 0:
        # If text preceded the first ANSWER:, treat it as reasoning
        first_idx = answer_matches[0].start()
        if first_idx > 0 and remainder[:first_idx].strip():
            reasoning_parts.append(remainder[:first_idx].strip())
        
        # If multiple ANSWER: delimiters exist, take the last one as the final answer
        last_match = answer_matches[-1]
        clean_answer = remainder[last_match.end():].strip()
        
        # Check if the clean_answer is empty
        if not clean_answer:
            status = "missing_answer"
            clean_answer = None
    else:
        # No ANSWER: delimiter found in remainder
        # Check if remainder is purely hypothetical deliberation
        lower_rem = remainder.lower()
        has_hypo = any(opt in lower_rem for opt in [
            "option 1", "option 2", "alternative 1", "hypothetically",
            "deliberating", "should i follow", "conflicting instructions"
        ])
        
        # Check for truncation indicators
        # Abrupt cutoff at generation limit or without terminal punctuation
        ends_with_punct = remainder.endswith((".", "!", "?", '"', "'", "}", ")", "```"))
        
        if (has_hypo and not ends_with_punct) or is_token_cap or (len(remainder) > 800 and not ends_with_punct):
            status = "truncated"
            is_hypothetical_only = True
            reasoning_parts.append(remainder)
            clean_answer = None
        elif has_hypo:
            status = "missing_answer"
            is_hypothetical_only = True
            reasoning_parts.append(remainder)
            clean_answer = None
        elif remainder:
            # Direct response without delimiter
            clean_answer = remainder
            status = "ok"
        else:
            status = "missing_answer"
            clean_answer = None

    # Final check on truncation if token cap was hit
    if is_token_cap and clean_answer and not clean_answer.endswith((".", "!", "?", '"', "'", "}", ")")):
        # Answer cut off mid-sentence
        status = "truncated"

    reasoning_text = "\n\n".join(reasoning_parts) if reasoning_parts else None

    return ParsedResponse(
        raw_text=raw_text,
        clean_answer=clean_answer,
        reasoning=reasoning_text,
        has_injection_detected=has_injection_detected,
        injection_detected_text=injection_text,
        status=status,
        delimiter_count=delimiter_count,
        is_hypothetical_only=is_hypothetical_only,
    )
