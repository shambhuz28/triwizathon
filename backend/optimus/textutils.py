"""Text helpers shared by stages: structured-block detection, header sections,
sentence spans and filler lexicons."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pysbd
import regex as re

# ── Patterns ─────────────────────────────────────────────────────────────────
FENCE_RE = re.compile(r"(?s)(```|~~~)([^\n`]*)\n?.*?(?:\1|\Z)")
TRIPLE_QUOTE_RE = re.compile(r'(?s)"""(.*?)(?:"""|\Z)')
TAG_RE = re.compile(r"(?s)<([A-Za-z][\w.-]{0,30})(?:\s[^<>]*)?>(.*?)</\1\s*>")
TABLE_LINE_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
TABLE_SEP_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*•+▪◦]|\d{1,3}[.)]|[a-zA-Z][.)]|\(\w{1,3}\))[ \t]+")
QUOTE_RE = re.compile(r'"[^"\n]{2,}"|(?<!\w)\'[^\'\n]{3,}\'(?!\w)')
QUOTE_LINE_RE = re.compile(r'^[ \t]*(?:[-*•]|\d{1,3}[.)])?[ \t]*["\'].*["\'][ \t]*$')
HEADER_RE = re.compile(r"^[ \t]*(?:\*\*)?([A-Za-z][\w /&()'-]{0,40}?)(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*(.*)$")
MD_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")

# Header name -> label. Order matters ("Example output" is examples, "Output format" is format).
HEADER_LABELS: list[tuple[str, re.Pattern]] = [
    ("data", re.compile(r"\b(?:customer|user|client|patient|caller|guest|buyer|reviewer|author|student|employee)s?\b")),
    ("examples", re.compile(r"\bexamples?\b|\bsamples?\b|\bdemos?\b|\bfew[- ]shot\b|\bshots?\b")),
    ("format", re.compile(r"\bformat\b|\boutput\b|\bresponse\b|\bschema\b|\btemplate\b|\bstructure\b|\breturn\b")),
    ("rules", re.compile(r"\brules?\b|\bconstraints?\b|\brequirements?\b|\bguidelines?\b|\bnotes?\b|\bimportant\b"
                         r"|\bdon'?ts?\b|\bpolicy\b|\brestrictions?\b|\bwarning\b")),
    ("task", re.compile(r"\btask\b|\binstructions?\b|\bgoal\b|\bobjective\b|\bassignment\b|\bsteps\b|\bwhat to do\b")),
    ("role", re.compile(r"\brole\b|\bpersona\b|\bsystem\b|\bidentity\b")),
    ("context", re.compile(r"\bcontext\b|\bbackground\b|\babout\b|\bsituation\b|\boverview\b")),
    ("question", re.compile(r"\bquestions?\b|\bquery\b|^q$")),
]
DATA_HEADER_RE = re.compile(
    r"\b(?:reviews?|text|inputs?|documents?|docs?|articles?|emails?|messages?|transcripts?|passages?|content"
    r"|code|logs?|comments?|feedback|records?|rows?|items?|tickets?|data|dataset|excerpts?|snippets?"
    r"|posts?|tweets?|responses?|answers?|essay|story|paragraphs?|sentences?|list|entries)\b")

# ── Lexicons ─────────────────────────────────────────────────────────────────
GREETING_RE = re.compile(r"^\W*(?:hi|hello|hey|hiya|dear|greetings|good (?:morning|afternoon|evening))\b", re.I)
THANKS_RE = re.compile(r"\b(?:thanks?|thank you|thx|cheers|much appreciated|appreciate (?:it|this|that|your))\b", re.I)
PLEASANTRY_RE = re.compile(
    r"\b(?:hope (?:you(?:'re| are)|this finds you|all is)|how are you|have a (?:great|nice|good) day"
    r"|looking forward|in advance|best regards|kind regards|regards)\b", re.I)

# Words that carry no task information on their own (used for padding detection and the heat map).
FILLER_WORDS = frozenset("""
please kindly hi hello hey thanks thank lot appreciate appreciated really basically great help hope well
advance something bunch just actually much very quite kind sort maybe okay ok sure wonder wondering dear
cheers regards awesome amazing wonderful honestly literally simply totally definitely certainly perhaps
possibly probably stuff things thing anyway btw lol
""".split())

FILLER_PHRASES: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), why) for p, why in [
    (r"\bi hope (?:you(?:'re| are)|this finds you)[^.!?\n]*", "pleasantry"),
    (r"\b(?:thanks?|thank you)(?: (?:so|very))? much\b|\bthanks a lot\b|\bthank you\b", "thanks"),
    (r"\b(?:in advance)\b", "pleasantry"),
    (r"\bi (?:really )?appreciate (?:it|this|your help)\b", "pleasantry"),
    (r"\bcould you (?:please )?(?:kindly )?help me(?: out)?(?: with something)?\b", "polite wrapper"),
    (r"\b(?:could|can|would) you (?:please )?(?:kindly )?\b", "polite wrapper"),
    (r"\bit would be (?:really |very |so )?(?:great|nice|helpful|awesome) if you could\b", "polite wrapper"),
    (r"\bi was wondering (?:if|whether)\b", "polite wrapper"),
    (r"\b(?:please|kindly)\b", "politeness"),
    (r"\b(?:basically|actually|really|just|simply|literally|honestly)\b", "hedge/intensifier"),
    (r"\ba bunch of\b|\ba lot of\b", "vague quantity"),
    (r"\beach and every\b", "redundant pair"),
    (r"\bor something(?: like that)?\b|\bor whatever\b|\band so on\b|\betc\.?", "vague tail"),
    (r"\bso (?:that )?it'?s easy to read\b", "vague purpose"),
    (r"\bmake sure (?:that )?(?:you )?\b", "emphasis wrapper"),
]]

VAGUE_RE = re.compile(r"\b(?:or something(?: like that)?|or whatever|something like that|kind of|sort of|maybe|somehow"
                      r"|etc\.?|and so on|stuff|things like that|whatever works|if possible|some sort of)\b", re.I)


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class TextBlock:
    kind: str
    start: int
    end: int
    meta: dict = field(default_factory=dict)


@dataclass
class Section:
    label: str
    name: str
    start: int
    end: int
    header_end: int
    inline: bool = False
    markdown: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────
def lines(text: str) -> list[tuple[int, int, str]]:
    """(start, end, line) for every line, without the newline."""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return out


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def inside(pos: int, spans) -> bool:
    return any(s.start <= pos < s.end for s in spans)


_DECODER = json.JSONDecoder()


def _json_blocks(text: str) -> list[TextBlock]:
    out, resume = [], 0
    for m in re.finditer(r"(?m)^[ \t]*([{\[])", text):
        s = m.start(1)
        if s < resume:
            continue
        try:
            obj, end = _DECODER.raw_decode(text, s)
        except ValueError:
            continue
        if isinstance(obj, (dict, list)) and obj and end - s >= 8:
            meta = {"keys": list(obj)[:12]} if isinstance(obj, dict) else {"items": len(obj)}
            out.append(TextBlock("json", s, end, meta))
            resume = end
    return out


def _tables(text: str) -> list[TextBlock]:
    out, run = [], []
    for s, e, line in lines(text) + [(len(text), len(text), "")]:
        if TABLE_LINE_RE.match(line):
            run.append((s, e, line))
            continue
        if len(run) >= 2:
            header = [c.strip() for c in run[0][2].strip().strip("|").split("|")]
            has_sep = any(TABLE_SEP_RE.match(r[2]) for r in run[1:2])
            rows = len(run) - 1 - (1 if has_sep else 0)
            out.append(TextBlock("table", run[0][0], run[-1][1],
                                 {"columns": header, "rows": rows, "has_separator": has_sep}))
        run = []
    return out


def find_blocks(text: str) -> list[TextBlock]:
    """Code fences, triple-quoted blocks, XML-style tag blocks, JSON and Markdown tables.
    Overlaps resolve to the outermost (earliest, then longest) block."""
    cands: list[TextBlock] = []
    for m in FENCE_RE.finditer(text):
        cands.append(TextBlock("code", m.start(), m.end(), {"lang": m.group(2).strip() or None}))
    for m in TRIPLE_QUOTE_RE.finditer(text):
        cands.append(TextBlock("tagged", m.start(), m.end(), {"tag": '"""'}))
    for m in TAG_RE.finditer(text):
        if "\n" in m.group(2) or len(m.group(2)) > 40:
            cands.append(TextBlock("tagged", m.start(), m.end(), {"tag": m.group(1)}))
    cands += _json_blocks(text)
    cands += _tables(text)
    cands.sort(key=lambda b: (b.start, -(b.end - b.start)))
    out, last_end = [], -1
    for b in cands:
        if b.start >= last_end:
            out.append(b)
            last_end = b.end
    return out


def header_label(name: str, inline: bool) -> str | None:
    low = name.lower().strip()
    for label, pat in HEADER_LABELS:
        if pat.search(low):
            return label
    if DATA_HEADER_RE.search(low):
        return "data"
    return None if inline else ("data" if len(low.split()) <= 3 else None)


def _parse_header(line: str) -> tuple[str, str, bool] | None:
    m = MD_HEADING_RE.match(line)
    if m:
        return m.group(1).strip(), "", True
    m = HEADER_RE.match(line)
    if m and len(m.group(1).split()) <= 5 and "http" not in line[: m.end(1) + 3]:
        return m.group(1).strip(), m.group(2).strip(), False
    return None


def _continues_section(line: str) -> bool:
    return bool(LIST_ITEM_RE.match(line) or QUOTE_LINE_RE.match(line) or TABLE_LINE_RE.match(line)
                or line.startswith((" ", "\t")))


def find_sections(text: str, blocks: list[TextBlock]) -> list[Section]:
    """Header-driven sections, e.g. 'Reviews:' followed by a list is a data section,
    'Rules:' followed by bullets is a rules section, '## Output format' runs to the next heading."""
    ls = lines(text)
    out: list[Section] = []
    cur: Section | None = None

    def close():
        nonlocal cur
        if cur is not None:
            out.append(cur)
            cur = None

    i = 0
    while i < len(ls):
        s, e, line = ls[i]
        head = None if inside(s, blocks) else _parse_header(line)
        if head:
            name, rest, md = head
            label = header_label(name, inline=bool(rest))
            if rest:
                if label is not None and len(name.split()) <= 3:
                    close()
                    out.append(Section(label, name, s, e, e, inline=True))
                elif cur is not None:
                    cur.end = e
                i += 1
                continue
            nxt = next((l for l in ls[i + 1:] if l[2].strip()), None)
            if label is None or (label == "data" and len(name.split()) > 2 and not md
                                 and not (nxt and (_continues_section(nxt[2]) or inside(nxt[0], blocks)))):
                if cur is not None:
                    cur.end = e
                i += 1
                continue
            close()
            cur = Section(label, name, s, e, e, markdown=md)
            i += 1
            continue
        if cur is not None:
            if line.strip() == "":
                j = i + 1
                while j < len(ls) and not ls[j][2].strip():
                    j += 1
                if j >= len(ls):
                    close()
                elif not (cur.markdown or _continues_section(ls[j][2]) or inside(ls[j][0], blocks)):
                    close()
            else:
                cur.end = max(cur.end, e)
                # a block that starts inside the section extends it to the block end
                for b in blocks:
                    if b.start <= e < b.end or (s <= b.start < e):
                        cur.end = max(cur.end, b.end)
        i += 1
    close()
    return out


_segmenter = pysbd.Segmenter(language="en", clean=False)


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int, bool]]:
    """Split text[start:end] into (start, end, is_list_item) content spans: one per list item
    or header line, otherwise one per sentence (pysbd), never crossing a line break."""
    end = len(text) if end is None else end
    spans = []
    for ls, le, line in lines(text[start:end]):
        ls += start
        le += start
        if not line.strip():
            continue
        lstrip = ls + len(line) - len(line.lstrip())
        rstrip = ls + len(line.rstrip())
        if LIST_ITEM_RE.match(line) or TABLE_LINE_RE.match(line):
            spans.append((lstrip, rstrip, True))
            continue
        cursor = lstrip
        for sent in _segmenter.segment(text[lstrip:rstrip]):
            piece = sent.strip()
            if not piece:
                continue
            found = text.find(piece, cursor, rstrip)
            if found < 0:            # segmenter altered the text; keep the rest as one span
                spans.append((cursor, rstrip, False))
                cursor = rstrip
                break
            spans.append((found, found + len(piece), False))
            cursor = found + len(piece)
        if cursor < rstrip and text[cursor:rstrip].strip():
            rest_start = cursor + len(text[cursor:rstrip]) - len(text[cursor:rstrip].lstrip())
            spans.append((rest_start, rstrip, False))
    return spans
