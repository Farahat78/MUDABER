from __future__ import annotations
import re

_CATEGORY_MAP: dict[str, list[str]] = {
    "dairy":      ["البان","ألبان","الألبان","لبن","حليب","جبن","جبنة","زبادي",
                   "dairy","milk","cheese","yogurt"],
    "proteins":   ["بروتين","فراخ","دجاج","لحم","سمك","بيض",
                   "protein","chicken","meat","fish","eggs","poultry"],
    "vegetables": ["خضار","خضروات","vegetables","veggies","veggie"],
    "fruits":     ["فاكهة","فواكه","fruit","fruits"],
    "grains":     ["نشويات","ارز","أرز","مكرونة","خبز",
                   "grains","rice","pasta","bread","carbs"],
    "beverages":  ["مشروبات","عصير","شاي","قهوة","مياه",
                   "beverages","drinks","juice","tea","coffee","water"],
    "snacks":     ["سناكس","شيبس","بسكويت","حلويات",
                   "snacks","chips","biscuits","sweets"],
    "cleaning":   ["تنظيف","منظفات","صابون","شامبو",
                   "cleaning","detergent","soap","shampoo"],
    "oils":       ["زيت","سمن","oils","butter","oil"],
    "spices":     ["توابل","بهارات","spices","condiments"],
}

_CAT_NAMES = {
    "البان","الالبان","الألبان","ألبان","dairy",
    "البروتين","البروتينات","proteins","protein",
    "خضار","خضروات","vegetables","الخضار",
    "فاكهة","فواكه","fruits","fruit","الفاكهة",
    "نشويات","grains","النشويات","carbs",
    "مشروبات","beverages","drinks","المشروبات",
    "سناكس","snacks","السناكس",
    "تنظيف","cleaning","منظفات",
    "زيوت","oils","spices","التوابل",
}

_REMOVE_KW = [
    "شيل","امسح","احذف","مش عايز","مش محتاج","بلاش","اشيل","ازيل",
    "remove","delete","without","don't want","dont want","take out","get rid",
]
_REPLACE_KW = [
    "غير","بدل","استبدل","اغير","بدلها","بدل نوع","change","replace","swap","switch",
]
_KEEP_PREFIXES = [
    "مش عايز غير","مش عايز إلا","مش عايز الا","عايز بس","عايز فقط",
    "ابقي بس","ابقي فقط","خلي بس","خلي فقط","خليها بس","خليه فقط",
    "keep only","i only want","i want only","only keep","just keep","احتفظ",
]
_INCREASE_KW = [
    "زود","زيد","ضيف","اضف","أضف","زيادة","كتر",
    "add more","more","increase","double","ضاعف",
]
_DECREASE_KW = [
    "قلل","نقص","اقلل","قليل","reduce","less","decrease","fewer",
]

_STRUCTURAL_NOISE = {
    "مش","في","من","على","هو","هي","الا","إلا","غير","و",
    "and","or","the","a","an","i","please","كل","كلها","كله",
    "no","not","just",
}


def _n(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"ة", "ه", t)
    t = re.sub(r"ـ", "", t)
    t = re.sub(r"[\u064B-\u065F]", "", t)
    return t


def _has(text: str, kws: list[str]) -> bool:
    t = _n(text)
    return any(_n(k) in t for k in kws)


def _detect_cat(text: str) -> str | None:
    t = _n(text)
    for cat, kws in _CATEGORY_MAP.items():
        for kw in kws:
            if _n(kw) in t:
                return cat
    return None


def _extract_allowed(text: str) -> list[str]:
    t = _n(text)
    after = text
    for prefix in sorted(_KEEP_PREFIXES, key=len, reverse=True):
        idx = t.find(_n(prefix))
        if idx != -1:
            after = text[idx + len(prefix):].strip()
            t     = _n(after)
            break
    for kw in ["غير","إلا","الا","except","only","بس","فقط","just","in"]:
        kn  = _n(kw)
        idx = t.find(kn)
        if idx != -1:
            candidate = after[idx + len(kw):].strip()
            if candidate:
                after = candidate
                t     = _n(after)
                break
    for cat_name in sorted(_CAT_NAMES, key=len, reverse=True):
        suffix = _n(cat_name)
        if t.endswith(suffix):
            t     = t[: -len(suffix)].strip()
            after = after[: len(after) - len(cat_name)].strip()
            break
    parts = re.split(r"[،,وand&+\s]+", t)
    result: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) <= 1:
            continue
        if p in _STRUCTURAL_NOISE:
            continue
        if p in {_n(c) for c in _CAT_NAMES}:
            continue
        result.append(p)
    return result


def _parse_one(seg: str) -> list[dict]:
    t = _n(seg)
    is_keep = _has(seg, _KEEP_PREFIXES) or (
        _has(seg, ["مش عايز", "don't want", "dont want"]) and
        _has(seg, ["غير", "إلا", "الا", "except", "only", "بس", "فقط"])
    )
    is_remove   = not is_keep and _has(seg, _REMOVE_KW)
    is_replace  = not is_keep and _has(seg, _REPLACE_KW)
    is_increase = _has(seg, _INCREASE_KW)
    is_decrease = _has(seg, _DECREASE_KW)
    cat = _detect_cat(seg)
    rules: list[dict] = []

    if is_keep:
        allowed = _extract_allowed(seg)
        rules.append({
            "action":           "keep_only",
            "category":         cat or "unknown",
            "allowed_keywords": allowed,
        })

    elif is_replace and cat:
        rules.append({"action": "replace_category", "category": cat})

    elif is_replace:
        clean = t
        for kw in _REPLACE_KW:
            clean = clean.replace(_n(kw), "").strip()
        if clean:
            rules.append({"action": "replace", "keyword": clean})

    elif is_remove:
        if cat:
            clean = t
            for kw in _REMOVE_KW:
                clean = clean.replace(_n(kw), "").strip()
            for nw in list(_STRUCTURAL_NOISE) + list(_CAT_NAMES):
                clean = re.sub(r'(?<!\w)' + re.escape(_n(nw)) + r'(?!\w)', '', clean).strip()
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean and clean != _n(cat):
                rules.append({"action": "remove", "keyword": clean})
            else:
                rules.append({"action": "replace_category", "category": cat})
        else:
            clean = t
            for kw in _REMOVE_KW:
                clean = clean.replace(_n(kw), "").strip()
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean:
                rules.append({"action": "remove", "keyword": clean})

    elif is_increase and cat:
        rules.append({"action": "increase", "category": cat})

    elif is_decrease and cat:
        rules.append({"action": "decrease", "category": cat})

    return rules


def parse_user_input(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    segments = re.split(
        r"[،,؛;]\s*|\s+(?:و|and|but|also|كمان|وكمان)\s+",
        text,
        flags=re.IGNORECASE,
    )
    segments = [s.strip() for s in segments if s.strip()]
    all_rules: list[dict] = []
    for seg in segments:
        all_rules.extend(_parse_one(seg))
    if not all_rules:
        all_rules.extend(_parse_one(text))
    return all_rules
