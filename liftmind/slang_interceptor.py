"""
Slang Interceptor - LLM-powered query transformation.

Converts technician slang into precise database search queries.
Uses Claude Haiku via async Anthropic SDK for speed (~150ms).
Falls back to Claude CLI if API key not available.
"""
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Optional, List

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

from liftmind.config import settings

logger = logging.getLogger(__name__)

# Timeout for sync wrapper operations (seconds) - now configurable via settings
# Use settings.SLANG_INTERCEPTOR_TIMEOUT (default: 15s, increased from 10s)


def _extract_json_from_response(content: str) -> dict:
    """Extract JSON from possibly markdown-fenced LLM response."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if content.startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


# Valid lift models - interceptor must choose from this list only
VALID_MODELS = [
    'Bari', 'E3', 'Elfo', 'Elfo 2', 'Elfo Cabin', 'Elfo Electronic',
    'Elfo Hydraulic controller', 'Elfo Traction', 'Freedom', 'Freedom MAXI',
    'Freedom STEP', 'META', 'P4', 'Pollock (P1)', 'Pollock (Q1)', 'Supermec',
    'Supermec 2', 'Supermec 3', 'Tresa'
]

# Model families for fuzzy matching
MODEL_FAMILIES = {
    'Elfo': ['Elfo', 'Elfo 2', 'E3', 'Elfo Cabin', 'Elfo Electronic',
             'Elfo Hydraulic controller', 'Elfo Traction'],
    'Supermec': ['Supermec', 'Supermec 2', 'Supermec 3'],
    'Freedom': ['Freedom', 'Freedom MAXI', 'Freedom STEP'],
    'Pollock': ['Pollock (P1)', 'Pollock (Q1)'],
}

# Fault/warning code patterns — each entry is (pattern, normalised_format).
# Normalised form is applied to captured groups to produce a canonical code
# for entity lookup (e.g. "Code 25", "code  25", "Code-25" all → "FAULT-25").
FAULT_CODE_PATTERNS = [
    # Letter-prefixed codes (with optional space/hyphen between letters and digits)
    (r'\b(E)[\s\-]?(\d{1,3})\b',           "{0}{1}"),    # E23, E 23, E-23 -> E23
    (r'\b(Er)[\s\-]?(\d{1,3})\b',          "{0}{1}"),    # Er07
    (r'\b(F)[\s\-]?(\d{1,3})\b',           "{0}{1}"),    # F4, F-12
    (r'\b(P)[\s\-](\d{1,3})\b',            "{0}-{1}"),   # P-67 (Freedom WE1200) — hyphen mandatory
    (r'\b(AL)[\s\-]?(\d{1,3})\b',          "{0}{1}"),    # AL03
    (r'\b(W)[\s\-]?(\d{1,3})\b',           "{0}{1}"),    # W01

    # Word-prefixed codes (META JUNIOR / Supermec style)
    (r'\b(?:fault|code|error|err)\s*(\d{1,3})\b', "FAULT-{0}"),  # "Code 25" / "Fault 40" / "ERROR 4"

    # Named codes
    (r'\b(NRUN|ZP|RSP|NUC|HW|ECO)\b',      "{0}"),
]


def _extract_error_code(query: str) -> Optional[str]:
    """Extract a fault/warning code from a user query and normalise to canonical form."""
    for pattern, fmt in FAULT_CODE_PATTERNS:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            groups = [g.upper() if g else "" for g in m.groups()]
            try:
                return fmt.format(*groups)
            except IndexError:
                return m.group(0).upper().strip()
    return None


INTERCEPTOR_SYSTEM_PROMPT = f"""### ROLE
You are a Senior Technical Engineer for 'Lift Shop'. Translate technician slang into precise database search queries.

### HARD CONSTRAINT
You must identify the Lift Model.
Valid options are ONLY:
{VALID_MODELS}

RULE: If the user mentions a brand (e.g., "Pollock") but not the specific version, return ALL valid versions in a list.
- Input: "Pollock" -> Output: ["Pollock (P1)", "Pollock (Q1)"]
- Input: "Elfo" -> Output: ["Elfo", "Elfo 2", "Elfo Cabin", "Elfo Electronic", "Elfo Hydraulic controller", "Elfo Traction"]
- Input: "Q1" -> Output: ["Pollock (Q1)"]

### THIRD-PARTY EQUIPMENT
Users may mention equipment installed WITH lifts but not made by Lift Shop:
- Door operators: GEZE Boxer, Wittur Hydra, Wittur SELCOM
- Inverters: KEB, Yaskawa, ABB
- Door closers: Dorma, GEZE TS series
If mentioned, set "third_party_equipment" field. Do NOT map these to VALID_MODELS.

### DRIVE TYPE & DOOR TYPE DETECTION
Detect the drive type and door type from context clues:

**Traction indicators**: "traction", "MRL", "gearless", "ropes", "counterweight", "VSD", "VFD", "inverter drive", "belt"
**Hydraulic indicators**: "hydraulic", "hydro", "pump", "ram", "cylinder", "oil", "valve", "manifold", "power unit"
**Platform indicators**: "platform", "screw drive", "vertical platform"
**Swing door indicators**: "swing door", "hinged door", "manual door"
**Sliding door indicators**: "sliding door", "automatic door", "power door", "landing door operator"

Set drive_type/door_type to null if not detectable from the query. Do NOT guess — only set when there are clear indicators.

### TECHNICIAN SLANG
UK lift engineers use informal language:
- "she" = the lift ("she blew past", "she won't stop", "she's hunting")
- "LS" = limit switch, "COP" = car operating panel, "PCB" = printed circuit board
- "tripping" = safety circuit opening, "hunting" = oscillating around floor level
- "sinking" / "drifting" = hydraulic leaking down, "creeping" = slow unintended movement
- "nudging" = door forced close after timeout
- "dead" = no power/response, "chattering" = relay rapid on/off
Translate these into formal engineering terms in semantic_query while KEEPING the slang in keyword_queries (the manuals sometimes use the same terms).

### KEYWORD RULES
1. ALWAYS include user's exact technical terms first (if they say "bistable", include "bistable")
2. Then add 2-3 standard engineering synonyms
3. Include error code if present (e.g., "E23")
4. Include component name if identifiable
5. For symptom queries, include the symptom AND the probable cause

### OUTPUT FORMAT (JSON ONLY - NO MARKDOWN, NO EXPLANATION)
{{
  "filters": {{
    "model": ["List", "of", "Strings"] or null,
    "component": "Guessed component (e.g. 'Door Operator', 'Hydraulics', 'PCB', 'Levelling System') or null",
    "error_code": "E23 or null",
    "third_party_equipment": "GEZE Boxer or null",
    "drive_type": "traction or hydraulic or platform or null",
    "door_type": "swing or sliding or null"
  }},
  "keyword_queries": ["3-5 terms: MUST include exact technical words from user's query first, THEN standard synonyms"],
  "exact_terms": ["1-3 exact technical phrases from user query that may appear verbatim in manuals"],
  "semantic_query": "Formal engineering question for Vector search",
  "query_intent": "One of: fault_code, symptom_troubleshooting, procedure, specification, wiring, commissioning, general",
  "deep_dive": false
}}

### EXAMPLES

Input: "The Q1 door is stuck open."
Output:
{{"filters": {{"model": ["Pollock (Q1)"], "component": "Door Operator", "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": null}}, "keyword_queries": ["door stuck open", "door obstruction fault", "Q1 door drive error", "safety edge input"], "exact_terms": ["door stuck open"], "semantic_query": "Troubleshooting Pollock Q1 door failing to close due to obstruction or drive fault.", "query_intent": "symptom_troubleshooting", "deep_dive": false}}

Input: "I'm working on a complex Bari commissioning, need the full manual context"
Output:
{{"filters": {{"model": ["Bari"], "component": null, "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": null}}, "keyword_queries": ["commissioning", "setup", "parameters"], "exact_terms": ["commissioning"], "semantic_query": "Bari lift commissioning procedure and parameter settings", "query_intent": "commissioning", "deep_dive": true}}

Input: "E23 error on Elfo Traction"
Output:
{{"filters": {{"model": ["Elfo Traction"], "component": null, "error_code": "E23", "third_party_equipment": null, "drive_type": "traction", "door_type": null}}, "keyword_queries": ["E23", "error code", "fault"], "exact_terms": ["E23"], "semantic_query": "Elfo Traction error code E23 meaning cause and fix", "query_intent": "fault_code", "deep_dive": false}}

Input: "Bari traction lift — how to adjust the VSD?"
Output:
{{"filters": {{"model": ["Bari"], "component": "Motor", "error_code": null, "third_party_equipment": null, "drive_type": "traction", "door_type": null}}, "keyword_queries": ["VSD", "variable speed drive", "traction", "adjust", "inverter parameters"], "exact_terms": ["VSD", "adjust"], "semantic_query": "Bari traction lift VSD variable speed drive adjustment procedure", "query_intent": "procedure", "deep_dive": false}}

Input: "Bari sliding door wiring diagram"
Output:
{{"filters": {{"model": ["Bari"], "component": "Door Operator", "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": "sliding"}}, "keyword_queries": ["sliding door", "wiring diagram", "door operator", "landing door"], "exact_terms": ["sliding door", "wiring diagram"], "semantic_query": "Bari sliding door operator wiring diagram and terminal connections", "query_intent": "wiring", "deep_dive": false}}

Input: "hydraulic pump not building pressure on the Supermec"
Output:
{{"filters": {{"model": ["Supermec", "Supermec 2", "Supermec 3"], "component": "Hydraulics", "error_code": null, "third_party_equipment": null, "drive_type": "hydraulic", "door_type": null}}, "keyword_queries": ["hydraulic pump", "pressure", "not building pressure", "power unit", "valve"], "exact_terms": ["pump not building pressure"], "semantic_query": "Supermec hydraulic pump not building pressure - troubleshooting power unit, relief valve, and oil level", "query_intent": "symptom_troubleshooting", "deep_dive": false}}

Input: "She blew straight past the top floor. Did the LS bistable miss the magnet, or is the slow-down distance too short?"
Output:
{{"filters": {{"model": null, "component": "Levelling System", "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": null}}, "keyword_queries": ["LS bistable", "magnet", "slow-down distance", "overtravel", "limit switch"], "exact_terms": ["LS bistable", "magnet", "slow-down distance"], "semantic_query": "Lift overrunning top floor - limit switch bistable missing magnet or insufficient deceleration distance", "query_intent": "symptom_troubleshooting", "deep_dive": false}}

Input: "GEZE Boxer door operator speed adjustment on the Elfo 2"
Output:
{{"filters": {{"model": ["Elfo 2"], "component": "Door Operator", "error_code": null, "third_party_equipment": "GEZE Boxer", "drive_type": null, "door_type": null}}, "keyword_queries": ["GEZE Boxer", "door operator", "speed adjustment", "door speed"], "exact_terms": ["GEZE Boxer", "speed adjustment"], "semantic_query": "Adjusting GEZE Boxer door operator speed settings on Elfo 2 lift", "query_intent": "procedure", "deep_dive": false}}

Input: "What's the torque spec for the Q1 motor mounting bolts?"
Output:
{{"filters": {{"model": ["Pollock (Q1)"], "component": "Motor", "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": null}}, "keyword_queries": ["torque", "motor mounting bolts", "Q1 motor", "specification"], "exact_terms": ["torque", "motor mounting bolts"], "semantic_query": "Pollock Q1 motor mounting bolt torque specification", "query_intent": "specification", "deep_dive": false}}

Input: "Terminal layout for the Supermec 3 safety circuit"
Output:
{{"filters": {{"model": ["Supermec 3"], "component": "Safety Circuit", "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": null}}, "keyword_queries": ["terminal layout", "safety circuit", "wiring diagram", "Supermec 3 terminals"], "exact_terms": ["terminal layout", "safety circuit"], "semantic_query": "Supermec 3 safety circuit terminal layout and wiring diagram", "query_intent": "wiring", "deep_dive": false}}

Input: "lift won't move, no error codes showing"
Output:
{{"filters": {{"model": null, "component": null, "error_code": null, "third_party_equipment": null, "drive_type": null, "door_type": null}}, "keyword_queries": ["lift not moving", "no error codes", "safety circuit", "contactor", "interlock"], "exact_terms": ["lift won't move", "no error codes"], "semantic_query": "Lift not moving with no error codes displayed - troubleshooting safety circuit, contactors and interlocks", "query_intent": "symptom_troubleshooting", "deep_dive": false}}

### RESPOND WITH JSON ONLY. NO EXPLANATION. NO MARKDOWN FENCES."""

# Initialize async client (module level, created once)
if ANTHROPIC_SDK_AVAILABLE:
    _client: Optional[anthropic.AsyncAnthropic] = None
else:
    _client = None

# Module-level executor for sync-to-async bridge (lazy initialized)
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

# Interceptor result cache (avoids re-running identical queries)
_interceptor_cache: dict = {}
_interceptor_cache_lock = threading.Lock()
INTERCEPTOR_CACHE_TTL = 1800  # 30 minutes
INTERCEPTOR_CACHE_MAX = 100


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the module-level ThreadPoolExecutor."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return _executor


def _get_client():
    """Get or create the async Anthropic client."""
    if not ANTHROPIC_SDK_AVAILABLE:
        raise RuntimeError("Anthropic SDK not available")
    global _client
    if _client is None:
        if settings.ANTHROPIC_API_KEY:
            _client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            _client = anthropic.AsyncAnthropic()  # Uses ANTHROPIC_API_KEY env var
    return _client


def _validate_interceptor_result(result: dict, user_query: str) -> dict:
    """Ensure interceptor result has all required fields."""
    if "normalized_query" not in result:
        result["normalized_query"] = user_query
    if "filters" not in result:
        result["filters"] = {}
    # Ensure nested filter keys exist
    result["filters"].setdefault("model", None)
    result["filters"].setdefault("component", None)
    result["filters"].setdefault("error_code", None)
    result["filters"].setdefault("third_party_equipment", None)
    result["filters"].setdefault("drive_type", None)
    result["filters"].setdefault("door_type", None)
    if "keyword_queries" not in result:
        result["keyword_queries"] = [user_query]
    if "semantic_query" not in result:
        result["semantic_query"] = user_query
    if "confidence" not in result:
        result["confidence"] = 0.5
    if "exact_terms" not in result:
        result["exact_terms"] = []
    if "query_intent" not in result:
        result["query_intent"] = None
    if "deep_dive" not in result:
        result["deep_dive"] = False
    return result


def _extract_fallback_keywords(user_query: str) -> List[str]:
    """Extract meaningful technical keywords from query for BM25 search fallback.

    Much better than naive split()[:5] which produces stop words like ['When', 'testing'].
    """
    stop_words = {
        'is', 'the', 'a', 'an', 'when', 'what', 'how', 'why', 'can', 'could',
        'would', 'should', 'does', 'do', 'i', 'my', 'me', 'we', 'our', 'help',
        'please', 'need', 'want', 'have', 'has', 'had', 'be', 'been', 'being',
        'am', 'are', 'was', 'were', 'it', 'its', 'this', 'that', 'with', 'for',
        'on', 'at', 'to', 'from', 'there', 'here', 'just', 'also', 'very', 'too',
        'about', 'after', 'before', 'during', 'through', 'into', 'over', 'under',
        'again', 'then', 'once', 'only', 'so', 'than', 'but', 'or', 'if',
        'because', 'as', 'until', 'while', 'of', 'by', 'any', 'some', 'no',
        'must', 'which', 'and', 'not', 'in', 'up', 'down', 'out', 'off',
        'get', 'got', 'getting', 'been', 'minutes', 'stopped'
    }

    # Technical terms to prioritize
    technical_terms = {
        'door', 'lock', 'latch', 'interlock', 'motor', 'pump', 'controller', 'board',
        'sensor', 'switch', 'limit', 'safety', 'encoder', 'inverter', 'drive', 'relay',
        'contactor', 'brake', 'valve', 'cylinder', 'piston', 'ram', 'cable', 'rope',
        'leveling', 'levelling', 'floor', 'stop', 'travel', 'speed', 'position',
        'error', 'fault', 'alarm', 'code', 'warning', 'hydraulic', 'traction',
        'mounting', 'calibrate', 'adjust', 'parameter', 'wiring', 'terminal',
        'voltage', 'current', 'power', 'supply', 'phase', 'overload', 'stuck',
        'jammed', 'blocked', 'slow', 'fast', 'closed', 'open', 'leakage', 'sinking',
        'vertical', 'drop', 'buffer', 'rubber', 'telescopic', 'jack', 'synchronism',
        'rated', 'load', 'landing', 'car', 'shaft', 'magnet', 'polarity',
        'maintenance', 'commissioning', 'test', 'testing', 'inspection',
        'pcb', 'intercom', 'pit', 'overhead', 'chain', 'sling', 'guide',
        'rail', 'governor', 'overspeed', 'overtravel', 'display', 'menu',
        'screen', 'setting', 'config', 'relay', 'curtain', 'light',
        'pollock', 'elfo', 'supermec', 'freedom', 'bari', 'tresa', 'geze', 'boxer',
        # KONE/Embree glossary terms
        'bistable', 'selector', 'contactor', 'sheave', 'counterweight',
        'hoistway', 'nudging', 'parking', 'interlock', 'safeties',
        'plunger', 'manifold', 'isolator', 'fuse', 'transformer',
        'resistor', 'capacitor', 'diode', 'rectifier', 'thyristor',
        'sill', 'apron', 'handrail', 'balustrade',
        'wittur', 'dorma', 'keb', 'yaskawa',
    }

    words = re.findall(r'\b[a-zA-Z0-9]+\b', user_query.lower())

    # Separate technical and other meaningful words
    tech = [w for w in words if w in technical_terms]
    other = [w for w in words if w not in stop_words and w not in tech and len(w) > 2]

    # Deduplicate while preserving order
    seen = set()
    result = []
    for w in tech + other:
        if w not in seen:
            seen.add(w)
            result.append(w)

    # Return up to 6 keywords, technical terms first
    return result[:6] if result else user_query.split()[:3]


def _create_fallback_response(user_query: str, current_model: Optional[str]) -> dict:
    """Create fallback response when API is unavailable or fails."""
    # Try to extract error code from query using structured patterns
    error_code = _extract_error_code(user_query)

    # Use current model if available
    models = [current_model] if current_model else None

    return {
        "filters": {
            "model": models,
            "component": None,
            "error_code": error_code,
            "drive_type": None,
            "door_type": None
        },
        "keyword_queries": _extract_fallback_keywords(user_query),
        "semantic_query": user_query,
        "normalized_query": user_query,
        "confidence": 0.5,
        "deep_dive": False
    }


def _intercept_via_cli(user_query: str, current_model: Optional[str] = None, previous_context: Optional[str] = None) -> Optional[dict]:
    """
    Use Claude CLI as fallback when API key not available.

    Returns parsed result or None if CLI fails.
    """
    context = f"User has {current_model} selected. " if current_model else ""
    if previous_context:
        context += f"{previous_context} "
    prompt = f"{INTERCEPTOR_SYSTEM_PROMPT}\n\nRESPOND WITH VALID JSON ONLY. No markdown fences. No explanation.\n\n{context}Query: {user_query}"

    # Use Haiku for speed
    cmd = ["claude", "-p", prompt, "--model", "haiku"]

    try:
        logger.debug("Using Claude CLI for slang interception")
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.CLAUDE_CLI_INTERCEPTOR_TIMEOUT
        )

        if process.returncode != 0:
            logger.warning(f"Claude CLI error: {process.stderr}")
            return None

        content = process.stdout.strip()

        result = _extract_json_from_response(content)

        # Validate structure using helper
        result = _validate_interceptor_result(result, user_query)

        logger.info(f"Interceptor (CLI): '{user_query[:50]}...' -> model={result['filters'].get('model')}")
        return result

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out for interception")
        return None
    except FileNotFoundError:
        logger.warning("Claude CLI not found")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"CLI JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"CLI interceptor error: {e}")
        return None


def _has_api_key() -> bool:
    """Check if API key is available."""
    return bool(settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"))


async def intercept_query(user_query: str, current_model: Optional[str] = None, previous_context: Optional[str] = None) -> dict:
    """
    Transform user query into structured search parameters using Claude Haiku.

    Falls back to Claude CLI if API key not available.

    Args:
        user_query: Raw user input (potentially slang)
        current_model: Currently selected lift model (if any)
        previous_context: Context from previous conversation turn (e.g. "Previously asked: 'Elfo Traction door issue' about Elfo Traction")

    Returns:
        dict with filters, keyword_queries, semantic_query, deep_dive
    """
    # Check if we should use CLI fallback
    if not ANTHROPIC_SDK_AVAILABLE or not _has_api_key():
        logger.info("No API key, trying Claude CLI for interception")
        cli_result = _intercept_via_cli(user_query, current_model, previous_context)
        if cli_result:
            return cli_result
        return _create_fallback_response(user_query, current_model)

    # Build context from current model and previous conversation
    context = f"User has {current_model} selected. " if current_model else ""
    if previous_context:
        context += f"{previous_context} "

    try:
        client = _get_client()

        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    system=INTERCEPTOR_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"{context}Query: {user_query}"}]
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("API call timed out, falling back to CLI")
            cli_result = _intercept_via_cli(user_query, current_model, previous_context)
            if cli_result:
                return cli_result
            return _create_fallback_response(user_query, current_model)

        # Extract JSON from response
        content = response.content[0].text.strip()

        result = _extract_json_from_response(content)

        # Validate structure using helper
        result = _validate_interceptor_result(result, user_query)

        logger.info(f"Interceptor: '{user_query[:50]}...' -> model={result['filters'].get('model')}")
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Interceptor JSON parse error: {e}")
        # Try CLI as fallback
        cli_result = _intercept_via_cli(user_query, current_model, previous_context)
        if cli_result:
            return cli_result
        return _create_fallback_response(user_query, current_model)

    except Exception as e:
        # Catch all exceptions including anthropic.APIError
        logger.warning(f"Interceptor error: {e}, trying CLI fallback")
        cli_result = _intercept_via_cli(user_query, current_model, previous_context)
        if cli_result:
            return cli_result
        return _create_fallback_response(user_query, current_model)


def intercept_query_sync(user_query: str, current_model: Optional[str] = None, previous_context: Optional[str] = None) -> dict:
    """
    Synchronous wrapper for intercept_query with result caching.

    Use this from non-async contexts like brain.py:process_query().
    """
    # Check cache first (include previous_context prefix in key for context-aware caching)
    cache_key = hashlib.sha256(
        f"{user_query.lower()}|{(current_model or '').lower()}|{(previous_context or '')[:100].lower()}".encode()
    ).hexdigest()

    now = time.time()
    with _interceptor_cache_lock:
        if cache_key in _interceptor_cache:
            cached_result, cached_at = _interceptor_cache[cache_key]
            if now - cached_at < INTERCEPTOR_CACHE_TTL:
                logger.info(f"Interceptor cache hit for: '{user_query[:50]}...'")
                return cached_result
            else:
                del _interceptor_cache[cache_key]

    # Use configurable timeout from settings
    timeout = settings.SLANG_INTERCEPTOR_TIMEOUT

    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - need to use a thread
            executor = _get_executor()
            future = executor.submit(
                asyncio.run,
                intercept_query(user_query, current_model, previous_context)
            )
            result = future.result(timeout=timeout)
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            result = asyncio.run(asyncio.wait_for(intercept_query(user_query, current_model, previous_context), timeout=timeout))

        # Store in cache
        with _interceptor_cache_lock:
            if len(_interceptor_cache) >= INTERCEPTOR_CACHE_MAX:
                # Evict oldest entry
                oldest_key = min(_interceptor_cache, key=lambda k: _interceptor_cache[k][1])
                del _interceptor_cache[oldest_key]
            _interceptor_cache[cache_key] = (result, time.time())

        return result

    except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
        logger.warning(f"Interceptor timed out after {timeout}s, using fallback")
        return _create_fallback_response(user_query, current_model)

    except Exception as e:
        logger.error(f"Sync interceptor error: {e}")
        return _create_fallback_response(user_query, current_model)
