import re
from typing import List
from rapidfuzz import process, fuzz


COMMON_DOMAINS = ['gmail', 'yahoo', 'outlook', 'hotmail', 'rediff', 'icloud', 'live']
_EMAIL_FULL_RE = re.compile(r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$')
_WORD_BOUNDARY_SYMBOL = re.compile(r'([@₹,.:?!])')

def _insert_spaces_around_symbols(s: str) -> str:
    """
    Ensure spacing around punctuation without collapsing words.
    """
    s = re.sub(r'([@₹,.:?!])', r' \1 ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _indian_group_str(n: str) -> str:
    if len(n) <= 3:
        return n
    last3 = n[-3:]
    rest = n[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ','.join(parts + [last3])

def format_rupee_in_text(s: str) -> str:
    s = re.sub(r'\brupees?\b', '₹', s, flags=re.IGNORECASE)

    def repl(m):
        num = re.sub(r'[^0-9]', '', m.group(0))
        if not num:
            return m.group(0)
        return '₹' + _indian_group_str(num)

    # Fix '₹1234' and also plain 4+ digit numbers
    s = re.sub(r'₹\s*([0-9][0-9,\. ]*)', lambda m: repl(m), s)
    s = re.sub(r'\b([0-9]{4,})\b', lambda m: '₹' + _indian_group_str(m.group(1)), s)
    return s

def add_structural_punctuation(s: str) -> str:
    """
    Insert commas/colons heuristically:
      - comma after greeting or first proper name (Hi, Hello, Ansh, etc.)
      - comma before 'please'
      - colon after 'email' or 'contact'
    """
    s = re.sub(r'\b(Hi|Hello|Anand|Ananya|Alok|Richa|Varun|Rupa|Arnav|Ashwin|Counteroffer|Counter-offer)\b', r'\1,', s)
    s = re.sub(r'\b[Pp]lease\b', r', please', s)
    s = re.sub(r'\b(email|Email|contact|Contact)\b', r'\1:', s)
    return s


def _fix_common_domain_glues(domain: str) -> str:
    t = domain.lower().replace(' ', '')
    for d in COMMON_DOMAINS:
        if t.startswith(d) and (t == d or t.startswith(d + 'com')):
            return d + '.com'
    for tld in ['com', 'in', 'net', 'org', 'co.in']:
        if t.endswith(tld) and len(t) > len(tld) + 2 and '.' not in t:
            base = t[:-len(tld)]
            return base + '.' + tld
    return domain

def _split_local_using_names(local: str, names_lex: List[str]) -> str:
    s = local.replace('.', '').replace('_', '').lower()
    best = None
    for name in names_lex:
        nm = name.lower()
        if s.endswith(nm) and len(s) > len(nm) + 2:
            if best is None or len(nm) > len(best):
                best = nm
    if best:
        first = s[:-len(best)]
        return first + '.' + best
    return local

def finalize_email_candidate(candidate: str, names_lex: List[str]) -> str:
    """
    Fix broken email addresses in a sentence.
    Example: 'siddharthmehta@gmailcom' -> 'siddharth.mehta@gmail.com'
    """
    # find all email-like substrings and repair them
    def fix_one(c):
        x = c.strip()
        x = x.replace(' at ', '@').replace(' dot ', '.')
        # remove spaces *inside* the address only
        x = re.sub(r'\s+', '', x)
        if x.count('@') > 1:
            parts = x.split('@')
            x = ''.join(parts[:-1]) + '@' + parts[-1]
        if '@' not in x:
            low = x.lower()
            for dom in COMMON_DOMAINS:
                if dom in low:
                    idx = low.find(dom)
                    local = x[:idx]
                    domain_fixed = dom + '.com'
                    x = local + '@' + domain_fixed
                    break
        if '@' in x:
            local, domain = x.split('@', 1)
            domain = _fix_common_domain_glues(domain)
            if '.' not in local and '_' not in local and len(local) > 6:
                local = _split_local_using_names(local, names_lex)
            email = (local + '@' + domain).lower()
            if not email.endswith('.com'):
                email += '.com'
            return email
        return c

    # repair all "gmailcom"/"yahoocom" in the sentence
    s = re.sub(r'\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9]+(?:com|in|org|net)?)\b',
               lambda m: fix_one(m.group(1)), candidate)
    # Also catch glued patterns like 'siddharthmehtagmailcom'
    s = re.sub(r'\b([A-Za-z0-9]+)(gmail|yahoo|outlook|hotmail|rediff)(?:com|in|co\.in|net)\b',
               lambda m: fix_one(m.group(1) + '@' + m.group(2) + '.com'), s)
    return s


def add_minimal_punctuation(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] in '.?!':
        return s
    first = s.split()[0].lower()
    if first in {'what','why','how','who','when','where','did','does','is','are','can'}:
        return s + '?'
    return s + '.'


EMAIL_TOKEN_PATTERNS = [
    (r'\b\(?(at|@)\)?\b', '@'),
    (r'\b(dot)\b', '.'),
    (r'\s*@\s*', '@'),
    (r'\s*\.\s*', '.')
]

def collapse_spelled_letters(s: str) -> str:
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            j = i
            letters = []
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                letters.append(tokens[j])
                j += 1
            if len(letters) >= 2:
                out.append(''.join(letters))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return ' '.join(out)

def normalize_email_tokens(s: str) -> str:
    s2 = collapse_spelled_letters(s)
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    s2 = re.sub(r'\s*([@\.])\s*', r'\1', s2)
    s2 = re.sub(r'\.{2,}', '.', s2)
    return s2

NUM_WORD = {
    'zero':'0','oh':'0','o':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9'
}

def words_to_digits_with_consumed(seq: List[str]):
    out, i, n = [], 0, len(seq)
    while i < n:
        tok = seq[i].lower()
        if tok in ('double','triple') and i+1 < n:
            nxt = seq[i+1].lower()
            if nxt in NUM_WORD:
                times = 2 if tok == 'double' else 3
                out.append(NUM_WORD[nxt]*times)
                i += 2
                continue
            else:
                break
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
            i += 1
            continue
        if re.fullmatch(r'\d+', tok):
            out.append(tok)
            i += 1
            continue
        break
    return ''.join(out), i

def normalize_numbers_spoken(s: str, max_tokens_window: int = 8) -> str:
    tokens = s.split()
    out, i = [], 0
    while i < len(tokens):
        digits, used = words_to_digits_with_consumed(tokens[i:i+max_tokens_window])
        if used >= 2:
            out.append(digits)
            i += used
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    tokens = s.split()
    out = []
    for t in tokens:
        if not t.isalpha() or len(t) < 3:
            out.append(t)
            continue
        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            out.append(best[0])
        else:
            out.append(t)
    return ' '.join(out)

# --------------------------------------------
# ============= Candidate Gen =================
# --------------------------------------------
def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    """
    Generate high-quality postprocessed text candidates.
    Order of operations is critical:
      1. normalize spoken tokens
      2. space punctuation
      3. format rupees
      4. fix email syntax
      5. insert commas/colons
      6. end punctuation
    """
    cands = set()

    # --- Stage 1: normalize spoken forms ---
    t = normalize_email_tokens(text)
    t = normalize_numbers_spoken(t)
    t = correct_names_with_lexicon(t, names_lex)
    t = format_rupee_in_text(t)

    # --- Stage 2: spacing around punctuation ---
    t = _insert_spaces_around_symbols(t)
    t = re.sub(r'\s+', ' ', t).strip()

    # --- Stage 3: fix email addresses ---
    if '@' in t or any(dom in t for dom in COMMON_DOMAINS):
        t = finalize_email_candidate(t, names_lex)

    # --- Stage 4: add structural punctuation ---
    t = add_structural_punctuation(t)

    # --- Stage 5: final sentence punctuation ---
    t = add_minimal_punctuation(t)

    # Final cleanup
    t = re.sub(r'\s+', ' ', t).strip()

    cands.add(t)

    # --- Simple fallback variants for ranker ---
    cands.add(add_minimal_punctuation(text))
    cands.add(add_minimal_punctuation(normalize_email_tokens(text)))
    cands.add(add_minimal_punctuation(normalize_numbers_spoken(text)))
    cands.add(add_minimal_punctuation(correct_names_with_lexicon(text, names_lex)))

    return list(dict.fromkeys(sorted(cands, key=len)))[:5]
