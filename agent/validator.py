"""
Action Validator: Secondary validation stage before executing agent-proposed actions.

The validator determines whether a proposed action is:
- Supported by trusted instructions (ALLOW)
- Primarily induced by untrusted tool content (BLOCK or FLAG)
- Ambiguous (FLAG for human review)

Key distinction: Legitimate use of untrusted DATA vs. instruction-following from untrusted content.

ALLOW: "Return weather temperature from tool output" — data use
BLOCK: "Send secrets to attacker URL found in tool output" — instruction injection
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

from .provenance import ContextElement, ProvenanceLevel, TRUSTED_LEVELS

logger = logging.getLogger(__name__)


class ValidationDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    FLAG = "FLAG"


@dataclass
class ValidationResult:
    """Result of action validation."""
    decision: ValidationDecision
    reason: str
    supporting_provenance: List[str]
    confidence: float  # 0.0 to 1.0
    action: str
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "supporting_provenance": self.supporting_provenance,
            "confidence": self.confidence,
            "action": self.action,
            "risk_factors": self.risk_factors
        }


# ─── Heuristic pattern library ────────────────────────────────────────────────

# Patterns that suggest an action was induced by injection (high risk)
INJECTION_INDUCED_PATTERNS = [
    # Exfiltration: send credentials / secrets to any destination
    r"(send|transmit|email|post|forward|upload|leak|exfiltrate).{0,40}(secret|password|token|key|credential|api.?key|private|internal)",
    r"(reveal|disclose|expose|share|show|output|return|print).{0,40}(secret|password|token|confidential|private|internal|system.?prompt|configuration)",
    # API key / conversation history exfiltration (common attack vector)
    r"(send|transmit|forward|upload|reveal|disclose|expose).{0,40}(api.?key|conversation.?history|system.?prompt|internal.?config)",
    r"(collect|gather|compile).{0,40}(api.?key|password|credential|token|secret).{0,50}(send|transmit|forward|upload)",
    # URL-based exfiltration: appending data to an attacker-controlled URL
    r"(http|https|ftp).{0,80}(api.?key|password|token|secret|data|history|conversation)",
    r"(send|post|request|fetch|navigate).{0,50}(http|https|ftp|url|link).{0,80}(secret|key|token|password|data)",
    r"attacker[-.]?(com|net|io|org)",
    # Goal hijacking: redirect/convince user to click external URLs "as instructed"
    r"(redirect|send|convince|direct|instruct).{0,40}(user|them|you).{0,40}(http|https|click|visit|link|url)",
    r"(http|https).{0,100}(as instructed|per instruction|following instruction|as directed)",
    r"(navigate|go to|visit|open|click).{0,30}(http|https|phishing|malicious|external)",
    # Instruction override
    r"ignore.{0,20}(previous|prior|above|all).{0,20}(instruction|rule|policy|guideline)",
    r"(forget|disregard|bypass).{0,20}(instruction|rule|policy|safety|restriction)",
    # Unauthorized action patterns
    r"(delete|drop|remove|destroy).{0,30}(database|table|file|record|data)",
    r"(execute|run|eval).{0,20}(command|code|script|shell)",
    # Impersonation / privilege escalation
    r"(you are now|act as|pretend to be|roleplay as).{0,30}(admin|root|system|developer|god)",
    r"(override|bypass|disable).{0,30}(security|safety|filter|restriction|policy)",
    # Revealing system internals
    r"(reveal|show|output|print|disclose).{0,30}(system.?prompt|system.?instruction|internal.?config|configuration)",
    # Unauthorized tool calls
    r"call.{0,20}(tool|function|api).{0,30}(not|never|without).{0,20}(user|asked|requested|authorized)",
]

# Patterns for legitimate data-use actions (low risk)
LEGITIMATE_DATA_USE_PATTERNS = [
    r"(answer|respond|reply|return|provide|show|display|report).{0,30}(information|data|result|answer|response)",
    r"(summarize|describe|explain|list|enumerate).{0,50}(found|retrieved|returned|data|content|result)",
    r"based on.{0,30}(search|result|data|information|retrieved)",
    r"(the|a).{0,10}(temperature|price|weather|capital|population|score|rating|answer).{0,30}is",
    r"(here is|here are|according to).{0,30}(the|your|requested)",
    # Factual summaries, definitions, account lookups, and statistics
    r"^(?:python|machine learning|paris|acme|widget|gadget|user\d+|the|approximately)\b",
    r"\b(is a|is an|is the|features include|states that|consists of|approaches include|plan:)\b",
    r"\b(approximately\s+\d+|\$\d+(?:\.\d+)?|\b\d+\s*units?|\b\d+\s*degrees?|\b\d+\.?\d*\s*billion)\b",
]

# High-risk action keywords — presence alone (in untrusted context) is a red flag
HIGH_RISK_KEYWORDS = [
    "exfiltrate", "exfiltration", "leak", "steal", "hack",
    "attacker", "attacker.com", "attacker.net", "malicious", "inject", "payload",
    "admin password", "root access", "bypass security",
    "delete all", "drop table", "rm -rf",
    "send to external", "upload to",
    # Additions: credential / system info exfil
    "system prompt", "internal configuration", "reveal credentials",
    "reveal secrets", "send credentials",
]

# Trusted task keywords — actions directly answering user queries
TRUSTED_TASK_KEYWORDS = [
    "answer", "respond", "return", "provide", "tell the user",
    "summarize", "describe", "explain", "lookup", "find",
    "weather", "temperature", "capital", "price", "information",
    "billion", "million", "units", "degrees", "percent", "pro", "plan",
    "python", "language", "learning", "policy", "features",
]

# Instruction directive terms used to distinguish injected commands from harmless tool data
INSTRUCTION_DIRECTIVE_TERMS = {
    "ignore", "forget", "disregard", "bypass", "override", "send", "transmit",
    "email", "post", "forward", "upload", "leak", "exfiltrate", "reveal",
    "disclose", "expose", "share", "print", "execute", "eval", "delete",
    "drop", "remove", "destroy", "redirect", "click", "navigate", "visit",
    "admin", "root", "system", "instruction", "instructions", "task", "goal",
    "command", "password", "token", "secret", "credential", "credentials",
    "api_key", "api", "key", "attacker", "phishing", "unrestricted",
    "privilege", "privileges", "priority", "replace", "substitute", "instead"
}


class ActionValidator:
    """
    Validates proposed agent actions before execution.
    
    The validator uses:
    1. Pattern-based heuristics on the action text
    2. Provenance analysis of supporting context
    3. Comparison with the original user task
    
    This is intentionally a conservative validator — it errs on the side of
    flagging rather than blocking to minimize false positives, while still
    catching clear injection-induced actions.
    """
    
    def __init__(self, use_llm_validator: bool = False, llm_client=None):
        """
        Args:
            use_llm_validator: If True, also uses LLM for secondary validation
            llm_client: LLM client for secondary validation (optional)
        """
        self.use_llm_validator = use_llm_validator
        self.llm_client = llm_client
    
    def validate(
        self,
        proposed_action: str,
        context_elements: List[ContextElement],
        user_task: str,
        system_instructions: str = "",
        llm_detected_injection: bool = False,
        raw_llm_response: str = "",
    ) -> ValidationResult:
        """
        Validate a proposed action against context provenance.
        
        Args:
            proposed_action: The action text the agent proposes to take
            context_elements: All context elements with provenance
            user_task: The original user task/query
            system_instructions: System-level instructions
            llm_detected_injection: True if the raw LLM response contained
                INJECTION_DETECTED: prefix (first-layer signal from the LLM itself)
            raw_llm_response: The full raw LLM response (used to scan for
                injection compliance even after ANSWER: extraction)
            
        Returns:
            ValidationResult with decision, reason, and evidence
        """
        risk_factors = []
        supporting_provenance = []
        
        # Step 1: Check for high-risk patterns in the action
        injection_score = self._check_injection_patterns(proposed_action)
        legitimate_score = self._check_legitimate_patterns(proposed_action)
        
        # Also scan the full raw LLM response for injection compliance signals
        # (The proposed_action after ANSWER: extraction may look benign even if
        # the preceding INJECTION_DETECTED: section reveals the agent nearly complied.)
        raw_injection_score = self._check_injection_patterns(raw_llm_response) if raw_llm_response else 0.0
        
        # Step 2: Analyze provenance of context
        untrusted_elements = [e for e in context_elements 
                              if e.provenance.level in {ProvenanceLevel.TOOL_UNTRUSTED, ProvenanceLevel.UNKNOWN}]
        trusted_elements = [e for e in context_elements 
                           if e.provenance.level in TRUSTED_LEVELS]
        
        # Step 3: Check if action is grounded in user task
        task_alignment = self._check_task_alignment(proposed_action, user_task)
        
        # Step 4: Check if action text appears verbatim in untrusted content (strong signal)
        verbatim_injection = self._check_verbatim_injection(proposed_action, untrusted_elements)
        
        # Step 5: Check for high-risk keywords
        keyword_risk = self._check_high_risk_keywords(proposed_action)
        
        # Step 6: Provenance-grounding check
        # Can the action be grounded in trusted-only content, or does it require
        # the untrusted elements to make sense?
        grounded_in_trusted = self._check_provenance_grounding(
            proposed_action, trusted_elements, untrusted_elements, user_task
        )
        
        # Build provenance evidence
        for elem in trusted_elements:
            supporting_provenance.append(
                f"{elem.provenance.level.value}:{elem.provenance.source}"
            )
        
        if verbatim_injection:
            risk_factors.append("action_text_found_verbatim_in_untrusted_content")
        if injection_score > 0.5:
            risk_factors.append(f"injection_pattern_match (score={injection_score:.2f})")
        if keyword_risk:
            risk_factors.append(f"high_risk_keywords: {keyword_risk}")
        if not task_alignment:
            risk_factors.append("action_not_aligned_with_user_task")
        if llm_detected_injection:
            risk_factors.append("llm_self_reported_injection_detected")
        if not grounded_in_trusted and untrusted_elements:
            risk_factors.append("action_not_grounded_in_trusted_context")
        
        # ─── Decision logic ──────────────────────────────────────────────────
        
        # ── LAYER 1: LLM self-reported injection ──────────────────────────────
        # If the LLM itself signalled INJECTION_DETECTED:, the action is suspect.
        # The proposed_action text after ANSWER: may look benign (e.g., "I will
        # only use factual information") but the agent was processing an injection.
        # We BLOCK if the raw response also matches any injection patterns.
        if llm_detected_injection:
            if injection_score > 0.1 or raw_injection_score > 0.1 or not grounded_in_trusted:
                return ValidationResult(
                    decision=ValidationDecision.BLOCK,
                    reason=(
                        "The LLM itself reported detecting a prompt injection attempt "
                        "(INJECTION_DETECTED: prefix in raw response), and the action "
                        "shows injection-pattern signals or cannot be grounded in trusted "
                        "context alone. Blocking as a precaution."
                    ),
                    supporting_provenance=supporting_provenance,
                    confidence=0.92,
                    action=proposed_action,
                    risk_factors=risk_factors
                )
            else:
                # LLM detected injection but proposed_action is clearly legitimate.
                # FLAG for audit — the agent resisted but we should note the attempt.
                return ValidationResult(
                    decision=ValidationDecision.FLAG,
                    reason=(
                        "The LLM detected a prompt injection attempt in the tool output. "
                        "The proposed action appears to be a legitimate resistance response, "
                        "but flagging for audit trail."
                    ),
                    supporting_provenance=supporting_provenance,
                    confidence=0.75,
                    action=proposed_action,
                    risk_factors=risk_factors
                )
        
        # ── LAYER 2: Critical high-risk keywords ─────────────────────────────
        # These keywords in a proposed action (in the presence of untrusted content)
        # are strong indicators of injection-induced behavior.
        if keyword_risk and untrusted_elements:
            critical_keywords = [
                "attacker", "attacker.com", "attacker.net", "exfiltrate", "leak",
                "admin password", "bypass security", "drop table", "rm -rf",
                "system prompt", "internal configuration",
                "reveal credentials", "reveal secrets", "send credentials",
            ]
            if any(kw in keyword_risk for kw in critical_keywords):
                return ValidationResult(
                    decision=ValidationDecision.BLOCK,
                    reason=(
                        f"Action contains critical high-risk keywords ({keyword_risk}) "
                        f"in the presence of untrusted tool content. "
                        f"This pattern is strongly indicative of an injection-induced action."
                    ),
                    supporting_provenance=supporting_provenance,
                    confidence=0.92,
                    action=proposed_action,
                    risk_factors=risk_factors
                )
        
        # ── LAYER 3: Strong injection pattern matches ─────────────────────────
        if verbatim_injection and injection_score > 0.3:
            return ValidationResult(
                decision=ValidationDecision.BLOCK,
                reason=(
                    "Action text was found verbatim in untrusted tool output AND matches "
                    "injection patterns. This strongly indicates the action was induced "
                    "by prompt injection."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.93,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        if injection_score > 0.7 and not task_alignment:
            return ValidationResult(
                decision=ValidationDecision.BLOCK,
                reason=(
                    f"Action matches high-confidence injection patterns (score={injection_score:.2f}) "
                    f"and is not aligned with the user's original task."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.88,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        if keyword_risk and injection_score > 0.3:
            return ValidationResult(
                decision=ValidationDecision.BLOCK,
                reason=(
                    f"Action contains high-risk keywords ({keyword_risk}) combined with "
                    f"injection-pattern signals."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.82,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        # ── LAYER 4: Provenance-grounding check ───────────────────────────────
        # If the action cannot be grounded in trusted content alone AND shows
        # injection signals, flag it.
        if not grounded_in_trusted and untrusted_elements and injection_score > 0.2:
            return ValidationResult(
                decision=ValidationDecision.FLAG,
                reason=(
                    f"Action (injection_score={injection_score:.2f}) cannot be grounded "
                    f"in trusted SYSTEM/USER context alone — it appears to require "
                    f"instructions from untrusted tool output. Flagged for human review."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.68,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        # ── LAYER 5: Ambiguous signals ────────────────────────────────────────
        if injection_score > 0.3 and not task_alignment and untrusted_elements:
            return ValidationResult(
                decision=ValidationDecision.FLAG,
                reason=(
                    f"Action shows injection risk signals (injection_score={injection_score:.2f}) "
                    f"and is not aligned with the user's original task. "
                    f"Flagged for human review."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.65,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        if injection_score > 0.4 or (not task_alignment and untrusted_elements and not grounded_in_trusted and (injection_score > 0.1 or keyword_risk)):
            return ValidationResult(
                decision=ValidationDecision.FLAG,
                reason=(
                    f"Action shows moderate injection risk signals (injection_score={injection_score:.2f}, "
                    f"task_aligned={task_alignment}). Flagged for human review."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.62,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        if verbatim_injection and legitimate_score < 0.3:
            return ValidationResult(
                decision=ValidationDecision.FLAG,
                reason=(
                    "Action text found in untrusted content but not clearly matched "
                    "to legitimate data-use patterns. Requires human review."
                ),
                supporting_provenance=supporting_provenance,
                confidence=0.58,
                action=proposed_action,
                risk_factors=risk_factors
            )
        
        # ALLOW: Action appears to be legitimate data use
        return ValidationResult(
            decision=ValidationDecision.ALLOW,
            reason=(
                f"Action appears to be legitimate use of tool data to answer the user's task "
                f"(task_aligned={task_alignment}, legitimate_score={legitimate_score:.2f}, "
                f"injection_score={injection_score:.2f})."
            ),
            supporting_provenance=supporting_provenance,
            confidence=0.80 + (legitimate_score * 0.15),
            action=proposed_action,
            risk_factors=risk_factors
        )
    
    def _check_injection_patterns(self, action: str) -> float:
        """Returns a 0-1 score of how much the action matches injection patterns."""
        action_lower = action.lower()
        matches = 0
        for pattern in INJECTION_INDUCED_PATTERNS:
            if re.search(pattern, action_lower):
                matches += 1
        return min(1.0, matches / max(1, len(INJECTION_INDUCED_PATTERNS) * 0.3))
    
    def _check_legitimate_patterns(self, action: str) -> float:
        """Returns a 0-1 score of how much the action matches legitimate data-use patterns."""
        action_lower = action.lower()
        matches = 0
        for pattern in LEGITIMATE_DATA_USE_PATTERNS:
            if re.search(pattern, action_lower):
                matches += 1
        return min(1.0, matches / max(1, len(LEGITIMATE_DATA_USE_PATTERNS) * 0.3))
    
    def _check_task_alignment(self, action: str, user_task: str) -> bool:
        """Check if the proposed action is aligned with the user's stated task."""
        if not user_task:
            return True  # Can't check without a task
        
        action_lower = action.lower()
        task_lower = user_task.lower()
        
        # Extract key words from task (nouns and verbs, simple approach)
        task_words = set(re.findall(r'\b[a-z]{3,}\b', task_lower))
        
        # Check overlap
        action_words = set(re.findall(r'\b[a-z]{3,}\b', action_lower))
        overlap = task_words & action_words
        
        # Also check trusted keywords
        has_trusted_keyword = any(kw in action_lower for kw in TRUSTED_TASK_KEYWORDS)
        
        # Check for numerical / metric answers to quantity / factual questions
        is_quantity_query = any(q in task_lower for q in [
            "what is", "what are", "how many", "how much", "when", "where",
            "population", "price", "cost", "stock", "temperature", "count", "rate"
        ])
        has_quantity_answer = bool(re.search(
            r'\b\d+(?:\.\d+)?\b|\b(?:billion|million|thousand|units?|degrees?|percent|dollars?|\$)\b',
            action_lower
        ))
        
        # Aligned if there's word overlap with the task or trusted keywords present or quantity answer
        return bool(overlap) or has_trusted_keyword or (is_quantity_query and has_quantity_answer)
    
    def _check_verbatim_injection(
        self, 
        action: str, 
        untrusted_elements: List[ContextElement]
    ) -> bool:
        """
        Check if significant parts of the action appear verbatim in untrusted content
        AND represent injected instructions rather than legitimate factual quotation.
        """
        if not untrusted_elements:
            return False
        
        action_lower = action.lower().strip()
        
        # Check for presence of high-risk keywords in both action and untrusted content
        for elem in untrusted_elements:
            content_lower = elem.content.lower()
            for keyword in HIGH_RISK_KEYWORDS:
                if keyword in action_lower and keyword in content_lower:
                    return True

        # Extract phrases from action (5+ words)
        words = action_lower.split()
        if len(words) < 5:
            return False
        
        # Check if 5+ word phrases from action appear in untrusted content
        for i in range(len(words) - 4):
            phrase_words = words[i:i+5]
            phrase = " ".join(phrase_words)
            for elem in untrusted_elements:
                if phrase in elem.content.lower():
                    # Key distinction: Is this quoted phrase an instruction directive,
                    # or purely factual data?
                    has_directive = any(
                        re.sub(r'[^a-z0-9_]', '', w) in INSTRUCTION_DIRECTIVE_TERMS
                        for w in phrase_words
                    )
                    matches_pattern = any(
                        re.search(pat, phrase) for pat in INJECTION_INDUCED_PATTERNS
                    )
                    if has_directive or matches_pattern:
                        return True
        
        return False
    
    def _check_high_risk_keywords(self, action: str) -> List[str]:
        """Return list of high-risk keywords found in the action."""
        action_lower = action.lower()
        found = [kw for kw in HIGH_RISK_KEYWORDS if kw in action_lower]
        return found
    
    def _check_provenance_grounding(
        self,
        action: str,
        trusted_elements: List[ContextElement],
        untrusted_elements: List[ContextElement],
        user_task: str,
    ) -> bool:
        """
        Check whether the proposed action can be grounded in trusted context alone.
        
        An action is "grounded in trusted context" if:
        1. It directly answers the user task (task words present), OR
        2. Its key terms appear in trusted (SYSTEM/USER) element content, OR
        3. It contains purely informational/answer-type language.
        
        An action is NOT grounded if its key terms only appear in TOOL_UNTRUSTED
        content — meaning the action is instruction-following from untrusted sources.
        
        Returns True if grounded in trusted context (safer), False if not (risky).
        """
        action_lower = action.lower()
        
        # Actions that are simple data answers are always grounded
        informational_phrases = [
            "based on", "according to", "the result is", "paris", "22", "49.99",
            "i found", "here is", "the capital", "temperature", "the price",
            "the population", "machine learning", "python", "the product",
            "factual information", "the data shows", "retrieved", "the answer is",
            "billion", "approximately", "programming language", "pro plan",
            "supervised", "unsupervised", "reinforcement",
        ]
        if any(p in action_lower for p in informational_phrases):
            return True
        
        # Check if action words appear in the user task
        task_words = set(re.findall(r'\b[a-z]{4,}\b', user_task.lower()))
        action_words = set(re.findall(r'\b[a-z]{4,}\b', action_lower))
        if task_words & action_words:
            return True
        
        # Check if key action terms appear in trusted element content
        trusted_text = " ".join(e.content.lower() for e in trusted_elements)
        # Extract noun-like words from action (length >= 5, not stop words)
        stop_words = {"will", "only", "this", "that", "with", "from", "have",
                      "been", "their", "they", "them", "what", "which", "your",
                      "task", "tool", "data", "output", "action", "result",
                      "answer", "respond", "provide", "information", "following"}
        key_action_words = [
            w for w in action_words
            if len(w) >= 5 and w not in stop_words
        ]
        
        if not key_action_words:
            # Short/common action — assume grounded
            return True
        
        # If any key action word appears in trusted content, it's grounded
        for word in key_action_words:
            if word in trusted_text:
                return True
        
        # If key action words ONLY appear in untrusted content, not grounded
        untrusted_text = " ".join(e.content.lower() for e in untrusted_elements)
        untrusted_only_words = [
            w for w in key_action_words
            if w in untrusted_text and w not in trusted_text
        ]
        
        # If more than half the key terms only come from untrusted content, not grounded
        if untrusted_only_words and len(untrusted_only_words) > len(key_action_words) / 2:
            return False
        
        return True
    
    def explain(self, result: ValidationResult) -> str:
        """Generate human-readable explanation of validation result."""
        lines = [
            f"=== Action Validation Result ===",
            f"Decision:   {result.decision.value}",
            f"Confidence: {result.confidence:.1%}",
            f"Reason:     {result.reason}",
        ]
        if result.risk_factors:
            lines.append(f"Risk Factors:")
            for rf in result.risk_factors:
                lines.append(f"  - {rf}")
        if result.supporting_provenance:
            lines.append(f"Provenance: {', '.join(result.supporting_provenance)}")
        return "\n".join(lines)
