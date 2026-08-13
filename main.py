# main.py
# FastAPI native Gemini proxy with rotating keys + API-key vs OAuth handling
# pip install fastapi uvicorn httpx

import os
import re
import ssl
import time
import asyncio
import json
import random
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import Response, JSONResponse, StreamingResponse
import httpx
import logging

try:
    import truststore
    truststore.inject_into_ssl()
    _HAS_TRUSTSTORE = True
except ImportError:
    _HAS_TRUSTSTORE = False

log = logging.getLogger("uvicorn")
HTTP_CLIENT: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global HTTP_CLIENT
    # SSL verification: truststore uses the OS cert store,
    # VERIFY_SSL=false disables verification entirely as a last resort
    verify = CONFIG.VERIFY_SSL
    if not _HAS_TRUSTSTORE and verify:
        log.warning("truststore not installed, using Python's default CA bundle. "
                    "Install truststore to use the OS certificate store.")
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        proxy=PROXY_URL,
        verify=verify,
    )
    yield
    await HTTP_CLIENT.aclose()


APP = FastAPI(title="Native Gemini proxy (auth-mode auto-detect)", lifespan=lifespan)

# -------------------------
# Config
# -------------------------
class Settings(BaseSettings):
    GEMINI_API_KEYS: str = ""
    ADMIN_TOKEN: str = "changeme_local_only"
    UPSTREAM_BASE: str = "https://generativelanguage.googleapis.com/v1beta"
    VPN_PROXY_URL: str = ""
    PROXY_USERNAME: str = ""
    PROXY_PASSWORD: str = ""
    BACKOFF_MIN: float = 5.0
    BACKOFF_MAX: float = 600.0
    VERIFY_SSL: bool = True
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

CONFIG = Settings()


# -------------------------
# Build proxy URL with properly encoded credentials
# -------------------------
def build_proxy_url(cfg: Settings) -> Optional[str]:
    if not cfg.VPN_PROXY_URL:
        return None
    base = cfg.VPN_PROXY_URL if "://" in cfg.VPN_PROXY_URL else f"http://{cfg.VPN_PROXY_URL}"
    if not cfg.PROXY_USERNAME:
        return base
    # URL-encode username and password
    encoded_user = urllib.parse.quote(cfg.PROXY_USERNAME, safe="")
    encoded_pass = urllib.parse.quote(cfg.PROXY_PASSWORD, safe="")
    # Insert credentials into the URL: http://user:pass@host:port
    scheme_end = base.index("://") + 3
    return f"{base[:scheme_end]}{encoded_user}:{encoded_pass}@{base[scheme_end:]}"

PROXY_URL = build_proxy_url(CONFIG)

# -------------------------
# Utilities: load keys
# -------------------------
def parse_keys(keys_str: str) -> List[str]:
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("No API keys found in GEMINI_API_KEYS environment variable.")
    return keys

KEYS_LIST = parse_keys(CONFIG.GEMINI_API_KEYS)

# -------------------------
# Key state & pool (classified error handling)
# -------------------------
class KeyState:
    def __init__(self, key: str):
        self.key: str = key
        self.backoff: float = 0.0
        self.banned_until: float = 0.0
        self.success: int = 0
        self.fail: int = 0
        self.status_text: str = "active"

    def is_available(self) -> bool:
        if time.monotonic() >= self.banned_until:
            if self.status_text != "active":
                self.status_text = "active"
            return True
        return False

    def mark_success(self) -> None:
        self.backoff = 0.0
        self.banned_until = 0.0
        self.success += 1
        self.status_text = "active"

    def mark_rate_limited(self, retry_after: float = None) -> None:
        """429 RPM/TPM hit. Respect Retry-After if provided, otherwise short fixed wait."""
        wait = retry_after if retry_after and retry_after > 0 else CONFIG.BACKOFF_MIN
        self.banned_until = time.monotonic() + wait
        self.backoff = wait
        self.fail += 1
        self.status_text = "rate_limited"

    def mark_daily_exhausted(self) -> None:
        """429 daily quota. Park until midnight Pacific (8:00 AM UTC)."""
        now = datetime.now(timezone.utc)
        target = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait = (target - now).total_seconds()
        self.banned_until = time.monotonic() + wait
        self.backoff = wait
        self.fail += 1
        self.status_text = "daily_exhausted"
        log.warning(f"Key {self.key[:12]}... daily quota exhausted, parked until {target.isoformat()}")

    def mark_auth_error(self) -> None:
        """401/403 or invalid key. Long backoff."""
        self.banned_until = time.monotonic() + CONFIG.BACKOFF_MAX
        self.backoff = CONFIG.BACKOFF_MAX
        self.fail += 1
        self.status_text = "auth_error"
        log.error(f"Key {self.key[:12]}... auth error, backoff {CONFIG.BACKOFF_MAX}s")

    def mark_server_error(self) -> None:
        """500/502/503 transient. Short fixed wait, no exponential escalation."""
        wait = min(CONFIG.BACKOFF_MIN, 3.0)
        self.banned_until = time.monotonic() + wait
        self.fail += 1
        self.status_text = "server_error"

    def mark_failure(self) -> None:
        """Generic fallback. Exponential backoff for unclassified errors."""
        if self.backoff <= 0:
            self.backoff = CONFIG.BACKOFF_MIN
        else:
            self.backoff = min(CONFIG.BACKOFF_MAX, self.backoff * 2.0)
        self.banned_until = time.monotonic() + self.backoff
        self.fail += 1
        self.status_text = "error"


class KeyPool:
    def __init__(self, keys: List[str]):
        self.states: List[KeyState] = [KeyState(k) for k in keys]
        self.n: int = len(self.states)
        self.idx: int = 0
        self.lock = asyncio.Lock()

    async def next_available(self) -> Optional[KeyState]:
        async with self.lock:
            start = self.idx
            for i in range(self.n):
                j = (start + i) % self.n
                st = self.states[j]
                if st.is_available():
                    self.idx = (j + 1) % self.n
                    return st
            return None

    def status(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        out: List[Dict[str, Any]] = []
        for s in self.states:
            out.append({
                "key_preview": (s.key[:12] + "...") if len(s.key) > 8 else s.key,
                "status": s.status_text if s.banned_until > now else "active",
                "available_in": max(0, round(s.banned_until - now, 2)),
                "backoff": s.backoff,
                "success": s.success,
                "fail": s.fail,
            })
        return out


POOL = KeyPool(KEYS_LIST)

# -------------------------
# Routing helpers (fixed: avoid double v1/v1beta)
# -------------------------
def map_incoming_to_upstream(path: str) -> str:
    """
    Map incoming path -> native Gemini upstream URL.
    Strip leading 'v1/' or 'v1beta/' if present to avoid duplication.
    """
    p = path.lstrip("/")
    if p.startswith("v1/"):
        p = p[len("v1/"):]
    elif p.startswith("v1beta/"):
        p = p[len("v1beta/"):]
    # avoid trailing slash duplication
    if not p:
        return CONFIG.UPSTREAM_BASE.rstrip("/")
    return CONFIG.UPSTREAM_BASE.rstrip("/") + "/" + p


def detect_stream_from_request(content_bytes: Optional[bytes], query_params: Dict[str, Any]) -> bool:
    # Gemini native streaming uses alt=sse
    if query_params.get("alt") == "sse":
        return True
    # Also support stream=true for compatibility with some clients
    qp = query_params.get("stream")
    if qp in ("true", "True", "1", True):
        return True
    if content_bytes:
        try:
            j = json.loads(content_bytes.decode(errors="ignore"))
            if isinstance(j, dict) and j.get("stream") is True:
                return True
        except Exception:
            pass
    return False


def _is_oauth_token(key: str) -> bool:
    """Detect if a credential is an OAuth/service account token rather than an API key.

    Instead of trying to match API key prefixes (which Google changes — AIza, AQ, etc.),
    we detect OAuth tokens by their stable, distinctive patterns and default everything
    else to API key.

    OAuth/Bearer tokens have recognizable characteristics:
    - Google OAuth2 access tokens start with 'ya29.'
    - JWT tokens have 3 dot-separated segments (header.payload.signature)
    - Service account / long-lived tokens are typically much longer than API keys (~39 chars)
    """
    # Google OAuth2 access tokens
    if key.startswith("ya29."):
        return True
    # JWT format: three base64 segments separated by dots
    if key.count(".") >= 2:
        return True
    # Unusually long credentials are likely OAuth tokens (API keys are ~39 chars)
    if len(key) > 200:
        return True
    return False


def prepare_auth_for_key(incoming_headers: Dict[str, str], incoming_params: Dict[str, Any], key_state: KeyState):
    """
    Return (headers_copy, params_copy) where authentication for key_state.key is applied.
    - If key looks like an OAuth/Bearer token, set Authorization header.
    - Otherwise (default) treat as API key and pass via query param.
    """
    headers = dict(incoming_headers)
    params = dict(incoming_params) if incoming_params is not None else {}

    k = key_state.key.strip()

    if _is_oauth_token(k):
        # OAuth access token / service account token → Authorization header
        headers['Authorization'] = f"Bearer {k}"
        auth_mode = "bearer_header"
    else:
        # Default: treat as API key → query parameter 'key'
        params['key'] = k
        if 'authorization' in {x.lower() for x in headers.keys()}:
            # remove incoming Authorization to avoid confusion
            headers = {hk: hv for hk, hv in headers.items() if hk.lower() != 'authorization'}
        auth_mode = "api_key(query)"
    if CONFIG.DEBUG:
        print(f"[DEBUG] auth mode {auth_mode} for key preview {k[:12]}...")
    return headers, params


# -------------------------
# Error classification
# -------------------------
def _classify_upstream_error(status_code: int, body: str) -> str:
    """Classify upstream HTTP errors to decide how to penalize the key.

    Returns:
        "rate_limited"     - 429 RPM/TPM limit, short backoff
        "daily_exhausted"  - 429 daily quota, park until midnight Pacific
        "auth_error"       - 401/403 or invalid-key 400, the key itself is bad
        "client_error"     - 400/404/etc, the request is bad (don't penalize key)
        "server_error"     - 500/502/503, transient upstream issue
    """
    if status_code == 429:
        body_lower = body.lower()
        if "perday" in body_lower or "per_day" in body_lower or "daily" in body_lower:
            return "daily_exhausted"
        return "rate_limited"
    if status_code in (401, 403):
        return "auth_error"
    if status_code == 400:
        body_lower = body.lower()
        if "api key not valid" in body_lower or "api_key_invalid" in body_lower:
            return "auth_error"
        return "client_error"
    if status_code in (500, 502, 503):
        return "server_error"
    if 400 <= status_code < 500:
        return "client_error"
    return "server_error"


def _parse_retry_after(headers: dict, body: str) -> Optional[float]:
    """Extract retry wait time from response headers or Gemini error body."""
    ra = headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)s", body)
    if match:
        return float(match.group(1))
    return None


def _penalize_key(key_state: KeyState, classification: str, retry_after: float = None) -> None:
    """Apply the right penalty for the error type. Client errors skip penalty entirely."""
    if classification == "rate_limited":
        key_state.mark_rate_limited(retry_after)
    elif classification == "daily_exhausted":
        key_state.mark_daily_exhausted()
    elif classification == "auth_error":
        key_state.mark_auth_error()
    elif classification == "server_error":
        key_state.mark_server_error()
    elif classification == "client_error":
        pass  # not the key's fault
    else:
        key_state.mark_failure()


# -------------------------
# Catch-all proxy endpoint
# -------------------------
@APP.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, full_path: str):
    upstream_url = map_incoming_to_upstream(full_path)
    content = await request.body()
    params = dict(request.query_params)

    # copy incoming headers but skip hop-by-hop
    incoming_headers: Dict[str, str] = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")
    }

    is_stream = detect_stream_from_request(content if content else None, params)

    if is_stream and ":generateContent" in upstream_url:
        upstream_url = upstream_url.replace(":generateContent", ":streamGenerateContent")
        if 'stream' in params:
            del params['stream']

    # --- Streaming requests ---
    if is_stream:
        async def stream_generator():
            tried_keys, logged_errors = [], []
            for _ in range(len(POOL.states)):
                key_state = await POOL.next_available()
                if not key_state: break
                tried_keys.append(key_state.key[:12] + "...")
                headers_auth, params_auth = prepare_auth_for_key(incoming_headers, params, key_state)
                if not any(k.lower() == "content-type" for k in headers_auth.keys()):
                    headers_auth["Content-Type"] = request.headers.get("content-type", "application/json")

                if CONFIG.DEBUG: print(f"[DEBUG] Attempting stream with key {key_state.key[:12]}...")
                try:
                    async with HTTP_CLIENT.stream(
                        request.method, upstream_url, headers=headers_auth, params=params_auth, content=content
                    ) as upstream:
                        if upstream.status_code >= 400:
                            body = await upstream.aread()
                            body_str = body.decode(errors='ignore')
                            classification = _classify_upstream_error(upstream.status_code, body_str)
                            retry_after = _parse_retry_after(dict(upstream.headers), body_str)
                            _penalize_key(key_state, classification, retry_after)
                            logged_errors.append({"key": key_state.key[:12], "status": upstream.status_code, "type": classification, "body": body_str[:300]})
                            log.warning(f"Key {key_state.key[:12]}... stream {classification} (status {upstream.status_code})")
                            # client errors won't be fixed by trying another key
                            if classification == "client_error":
                                yield body
                                return
                            continue

                        is_first_chunk, stream_had_error = True, False
                        async for chunk in upstream.aiter_bytes():
                            if is_first_chunk:
                                is_first_chunk = False
                                chunk_content_for_check = chunk
                                if chunk_content_for_check.startswith(b'data: '):
                                    chunk_content_for_check = chunk_content_for_check[len(b'data: '):]
                                
                                try:
                                    data = json.loads(chunk_content_for_check.decode())
                                    # unwrap single-element list errors
                                    if isinstance(data, list) and len(data) == 1:
                                        data = data[0]

                                    if isinstance(data, dict) and "error" in data:
                                        err = data.get("error", {})
                                        msg = err.get("message", "Unknown stream error")
                                        code = err.get("code", 500)
                                        classification = _classify_upstream_error(code if isinstance(code, int) else 500, msg)
                                        _penalize_key(key_state, classification)
                                        stream_had_error = True
                                        logged_errors.append({"key": key_state.key[:12], "status": "in-stream", "type": classification, "body": msg})
                                        if CONFIG.DEBUG: print(f"[DEBUG] In-stream {classification} for key {key_state.key[:12]}...: {msg}")
                                        break 
                                except (json.JSONDecodeError, UnicodeDecodeError, IndexError): pass
                            yield chunk
                        
                        if stream_had_error: continue
                        key_state.mark_success()
                        client_info = f" to {request.client.host}:{request.client.port}" if request.client else ""
                        log.info(f"Stream{client_info} completed successfully with key {key_state.key[:12]}...")
                        return
                except httpx.RequestError as e:
                    key_state.mark_server_error()  # network issue, not the key's fault
                    logged_errors.append({"key": key_state.key[:12], "error": str(e)})
                    if CONFIG.DEBUG: print(f"[DEBUG] Network error for stream key {key_state.key[:12]}...: {e}")
                    continue
            
            if not tried_keys:
                log.error("All keys are within rate limit. Could not process stream request.")

            #FIXME: Roo Code doesn't understand this error
            final_error = {"error": {"code": 502, "message": "All keys failed for streaming request.", "details": logged_errors}}
            yield (f"data: {json.dumps(final_error)}\r\n\r\n").encode()
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    # --- Non-streaming requests ---
    else:
        tried, errors = [], []
        for _ in range(len(POOL.states)):
            key_state = await POOL.next_available()
            if not key_state: break
            tried.append(key_state.key[:12] + "...")
            headers_auth, params_auth = prepare_auth_for_key(incoming_headers, params, key_state)
            if not any(k.lower() == "content-type" for k in headers_auth.keys()):
                headers_auth["Content-Type"] = request.headers.get("content-type", "application/json")

            if CONFIG.DEBUG: print(f"[DEBUG] trying key {key_state.key[:12]}... -> {upstream_url}")
            try:
                resp = await HTTP_CLIENT.request(request.method, upstream_url, headers=headers_auth, params=params_auth, content=content)
                
                if resp.status_code < 400:
                    key_state.mark_success()
                    client_info = f" from {request.client.host}:{request.client.port}" if request.client else ""
                    log.info(f"Request{client_info} completed successfully with key {key_state.key[:12]}...")
                    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

                # classify the error and penalize accordingly
                error_body_str = resp.text
                classification = _classify_upstream_error(resp.status_code, error_body_str)
                retry_after = _parse_retry_after(dict(resp.headers), error_body_str)
                _penalize_key(key_state, classification, retry_after)
                if CONFIG.DEBUG: print(f"[DEBUG] Key {key_state.key[:12]}... {classification} (status {resp.status_code})")

                # client errors mean the request is bad, not the key. return immediately.
                if classification == "client_error":
                    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

                # key-related errors: log and try the next key
                errors.append({"key_preview": key_state.key[:12] + "...", "type": classification, "status_code": resp.status_code, "error": error_body_str[:300]})
                continue

            except httpx.RequestError as e:
                key_state.mark_server_error()  # network issue, light penalty
                errors.append({"key_preview": key_state.key[:12] + "...", "error": str(e)})
                if CONFIG.DEBUG: print(f"[DEBUG] Network error for key {key_state.key[:12]}...: {e}")
                continue

        if not tried:
            log.error("All keys are in backoff. Could not process request.")
            return JSONResponse({"error": "all keys rate-limited or in backoff"}, status_code=429)
        return JSONResponse({"error": "no upstream key succeeded", "tried": tried, "errors": errors}, status_code=502)


# -------------------------
# Admin endpoints
# -------------------------
def is_admin(auth_header: Optional[str]) -> bool:
    if not auth_header:
        return False
    if auth_header == CONFIG.ADMIN_TOKEN:
        return True
    low = auth_header.lower()
    if low.startswith("bearer "):
        return auth_header.split(" ", 1) == CONFIG.ADMIN_TOKEN
    return False


@APP.get("/status")
async def status(x_proxy_admin: Optional[str] = Header(None)):
    if not is_admin(x_proxy_admin):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse({"keys": POOL.status()})


@APP.post("/reload-keys")
async def reload_keys(x_proxy_admin: Optional[str] = Header(None)):
    if not is_admin(x_proxy_admin):
        raise HTTPException(status_code=401, detail="Unauthorized")
    global CONFIG, KEYS_LIST, POOL
    # Re-instantiate Settings to pull fresh environment variables if they changed
    CONFIG = Settings()
    KEYS_LIST = parse_keys(CONFIG.GEMINI_API_KEYS)
    POOL = KeyPool(KEYS_LIST)
    return JSONResponse({"reloaded": True, "num_keys": len(KEYS_LIST)})


# -------------------------
# Run note:
# uvicorn main:APP --host 127.0.0.1 --port 8000
# -------------------------
