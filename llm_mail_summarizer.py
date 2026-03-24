import os
import re
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Tuple, Optional

# Initializations--------------------------------------------------------------------------------------------------------------------------
API_TOKEN   = os.getenv("API_TOKEN")
BIND_HOST   = os.getenv("BIND_HOST", "127.0.0.1") # 127.0.0.1
PORT        = int(os.getenv("PORT", "8010")) #8010
MAX_BODY_B  = int(os.getenv("MAX_BODY_BYTES", "1048576"))  # 2^20

# Split End of text
EOT_SPLIT_RE = re.compile(r"(?:<\|endoftext\|>|&lt;\|endoftext\|&gt;)", re.I)

# Pre training with heuristics---------------------------------------------------------------------------------------------------------
SENDER_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9_.’'`\- ]{0,63})\s*(?::|-|–|—)\s*(.+)$"
)

# Basic convo heuristics
WELCOME_PAT  = re.compile(r"\b(welcome|glad to have|great to have|congratulations)\b", re.I)
INTRO_PAT    = re.compile(r"\b(i\s*(am|'m)\s*new|joining|joined|new to (the )?team)\b", re.I)
DONE_PAT     = re.compile(r"\b(final( version)? sent|sent (the )?final|shared|submitted|done|completed|pushed|delivered)\b", re.I)
REQ_PAT      = re.compile(r"\b(can we|please|could you|do we|need to|request|help|action required)\b", re.I)
THANKS_PAT   = re.compile(r"\b(thanks|thank you|appreciate)\b", re.I)
QUESTION_PAT = re.compile(r"\?\s*$")

# Name
NAME_STOP = {"new", "team", "here", "everyone", "all", "folks", "there", "thanks", "hello", "hi"}

def guess_name_from_intro(original_text: str) -> Optional[str]:
    m = re.search(r"\b(?:This is|I am|I'm)\s+([A-Z][a-zA-Z_.\-]{1,63})\b", original_text)
    if m:
        cand = m.group(1).strip(" .,:;")
        if cand and cand.lower() not in NAME_STOP:
            return cand

    m2 = re.search(r"\bMy name is\s+([A-Z][a-zA-Z_.\-]{1,63})\b", original_text)
    if m2:
        cand = m2.group(1).strip(" .,:;")
        if cand and cand.lower() not in NAME_STOP:
            return cand

    return None

# Text normalization & parsing-----------------------------------------------------------------------------------------------
# Unicode
def normalize_punct(s: str) -> str:
    return (s.replace("：", ":")
             .replace("–", "-")
             .replace("—", "-")
             .replace("‒", "-")
             .replace("−", "-"))

# Clean ups
def clean_control(text: str) -> str:
    text = EOT_SPLIT_RE.sub(" ", text)
    return " ".join(ln.strip() for ln in text.splitlines() if ln.strip())

#Splits on senders for context
def split_segments(raw: str) -> List[Tuple[str, str]]:
    parts = [p.strip() for p in EOT_SPLIT_RE.split(raw)]
    segs: List[Tuple[str, str]] = []
    last_sender: Optional[str] = None

    for p in parts:
        if not p:
            continue
        p_norm = normalize_punct(p)
        m = SENDER_RE.match(p_norm)
        if m:
            sender = m.group(1).strip()
            sender = sender.strip(" <>")  # strip wrappers like <Aakriti>
            text = m.group(2).strip()
            last_sender = sender or last_sender
        else:
            # continuation of previous sender
            sender = last_sender if last_sender else "(unknown)"
            text = p_norm

        text = clean_control(text)
        if text:
            segs.append((sender, text))
    return segs

def compress_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def join_human(names: List[str]) -> str:
    if not names: return ""
    if len(names) == 1: return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"

# Finetuning------------------------------------------------------------------------------------------
def abstractive_summarize(raw: str, max_sentences: int = 3) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""

    segs = split_segments(raw)
    if not segs:
        return ""

    # Signals
    intro_subject: Optional[str] = None
    welcomes_from: List[str] = []
    actions: List[Tuple[str, str]] = []
    requests: List[Tuple[str, str]] = []
    thanks_from: List[str] = []
    participants: List[str] = []  # seen speakers with names

    for sender, text in segs:
        if sender not in participants and sender != "(unknown)":
            participants.append(sender)

        t = text.lower()

        if INTRO_PAT.search(t) and not intro_subject:
            if sender != "(unknown)":
                intro_subject = sender
            else:
                # case-sensitive guess (won't pick 'new')
                intro_subject = guess_name_from_intro(text)
                if not intro_subject and participants:
                    intro_subject = participants[0]

        if WELCOME_PAT.search(t):
            if sender not in welcomes_from and sender != "(unknown)":
                welcomes_from.append(sender)

        if DONE_PAT.search(t):
            actions.append((sender, text))

        if REQ_PAT.search(t) or QUESTION_PAT.search(text):
            requests.append((sender, text))

        if THANKS_PAT.search(t):
            if sender not in thanks_from and sender != "(unknown)":
                thanks_from.append(sender)

    sentences: List[str] = []

    # 1) Intro / context
    subject_nice = intro_subject or "the new teammate"
    if INTRO_PAT.search(" ".join([txt for _, txt in segs])):  # any intro signal present
        sentences.append(f"{subject_nice} is new to the team and introduced themselves")

    # 2) Welcomes
    welcomers = [w for w in welcomes_from if w != intro_subject]
    if welcomers:
        sentences.append(f"{join_human(welcomers)} welcomed {subject_nice}")

    # 3) Actions / confirmations
    if actions:
        a_sender, a_text = actions[-1]
        a_brief = re.sub(r"\s+", " ", a_text).strip()
        if len(a_brief) > 100:
            a_brief = a_brief[:97] + "..."
        sentences.append(f"{a_sender} confirmed completion/dispatch (e.g., {a_brief})")

    # 4) Latest request / question
    if requests:
        r_sender, r_text = requests[-1]
        r_brief = re.sub(r"\s+", " ", r_text).strip()
        if len(r_brief) > 100:
            r_brief = r_brief[:97] + "..."
        sentences.append(f"Current request/question from {r_sender}: \"{r_brief}\"")
    else:
        # If no explicit requests were detected but welcomes happened,
        # add a light closure sentence.
        if welcomers:
            sentences.append("No specific questions were raised in the thread")

    # Fallback if no clear signals
    if not sentences:
        first = compress_sentence(segs[0][1])
        last  = compress_sentence(segs[-1][1]) if len(segs) > 1 else ""
        summary = " ".join(s for s in [first, last] if s) or "Discussion noted."
        return summary

    # Tidy & cap
    summary = " ".join(compress_sentence(s) for s in sentences[:max_sentences]).strip()
    return summary or "No summary generated."

# HTTP format check----------------------------------------------------------------------------------------------------
def auth_ok(headers) -> bool:
    if not API_TOKEN:
        return True
    auth = headers.get("Authorization", "") or ""
    xkey = headers.get("x-api-key", "") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    elif xkey:
        token = xkey.strip()
    return token == API_TOKEN

def parse_body(req: BaseHTTPRequestHandler, max_len: int) -> dict:
    length = int(req.headers.get("Content-Length", "0"))
    if length > max_len:
        raise ValueError("Payload too large")
    raw = req.rfile.read(length) if length > 0 else b""
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise ValueError("Invalid JSON")

# HTTP handler--------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "MiniMailSummarizer/1.3"

    def _send(self, code: int, payload=None, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if payload is not None:
            self.wfile.write(json.dumps(payload).encode("utf-8"))
    
    def do_GET(self):
        if self.path == "/":
            return self._send(200, {"status": "running"})

        if self.path == "/healthz":
            return self._send(200, {"status": "ok"})

        return self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        try:
            if self.path not in ("/summarize"):
                return self._send(404, {"detail": "Not Found"})

            if not auth_ok(self.headers):
                return self._send(401, {"detail": "Unauthorized"})

            body = parse_body(self, MAX_BODY_B)

            # Build thread text from either shape
            email_text = ""
            if isinstance(body.get("messages"), list):
                parts: List[str] = []
                for m in body["messages"]:
                    c = (m or {}).get("content", "")
                    if isinstance(c, str) and c.strip():
                        parts.append(c.strip())
                email_text = "\n".join(parts).strip()
            if not email_text:
                email_text = str(body.get("text", "")).strip()

            if not email_text:
                return self._send(400, {"detail": "Missing 'text' or 'messages' content"})

            # Abstractive, delimiter-aware summary (allowing up to 3 short sentences)
            summary = abstractive_summarize(email_text, max_sentences=3) or "No summary generated."
            return self._send(200, {"summary": summary})
        
        except ValueError as ve:
            return self._send(400, {"detail": str(ve)})
        except Exception as e:
            return self._send(500, {"detail": f"Internal Server Error: {e.__class__.__name__}"})

# Entrypoint-------------------------------------------------------------------------------------------------------------
def run():
    httpd = HTTPServer((BIND_HOST, PORT), Handler)
    print(f"Serving on http://{BIND_HOST}:{PORT}", flush=True)
    print("Endpoints:\n  GET  /healthz\n  POST /summarize")
    if API_TOKEN:
        print("Auth: Authorization: Bearer <API_TOKEN>  (or x-api-key)")
    else:
        print("Auth: disabled (demo mode). Set API_TOKEN to enable.")
    httpd.serve_forever()

if __name__ == "__main__":
    run()