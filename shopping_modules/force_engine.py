from __future__ import annotations
import re

try:
    from rapidfuzz import fuzz, process as rfprocess
    _RF = True
except ImportError:
    _RF = False

_CONCEPTS: dict[str, list[str]] = {
    "chicken":    ["فراخ","دجاج","فرخه","اجنحه","صدور","ران","دجاجه"],
    "fish":       ["سمك","بلطي","بوري","تونا","سلمون","رنجه"],
    "meat":       ["لحم","بيف","لحمه"],
    "eggs":       ["بيض","بيضه"],
    "milk":       ["حليب","لبن"],
    "cheese":     ["جبنه","جبن"],
    "yogurt":     ["زبادي"],
    "rice":       ["ارز","رز"],
    "pasta":      ["مكرونه","باستا"],
    "bread":      ["خبز","توست","عيش"],
    "apple":      ["تفاح"],
    "banana":     ["موز"],
    "orange":     ["برتقال"],
    "mango":      ["مانجو"],
    "tomato":     ["طماطم","تماتم"],
    "cucumber":   ["خيار"],
    "chips":      ["شيبس","كريسبي"],
    "biscuits":   ["بسكويت"],
    "oil":        ["زيت"],
    "shampoo":    ["شامبو"],
    "soap":       ["صابون"],
    "juice":      ["عصير"],
    "water":      ["مياه","ميه"],
    "tea":        ["شاي"],
    "coffee":     ["قهوه","نسكافيه"],
}


def _norm(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"ة", "ه", t)
    t = re.sub(r"ـ", "", t)
    t = re.sub(r"[\u064B-\u065F]", "", t)
    return t


def _concept_synonyms(query: str) -> list[str]:
    q      = _norm(query)
    result = [q]
    for concept, synonyms in _CONCEPTS.items():
        if q == concept or q in [_norm(s) for s in synonyms]:
            result += [_norm(s) for s in synonyms]
            result.append(concept)
            break
    return list(dict.fromkeys(result))


def _has_valid_price(item: dict) -> bool:
    """An item is valid only if it carries a positive price."""
    price = item.get("unit_price") or item.get("effective_price") or item.get("price", 0)
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def fuzzy_match(query: str, choices: list[str], threshold: int = 55) -> str | None:
    if not choices or not query:
        return None
    q            = _norm(query)
    norm_choices = [_norm(c) for c in choices]
    if _RF:
        r = rfprocess.extractOne(q, norm_choices, scorer=fuzz.partial_ratio)
        if r and r[1] >= threshold:
            return choices[r[2]]
    else:
        for i, nc in enumerate(norm_choices):
            if q in nc or nc in q:
                return choices[i]
    return None


def fuzzy_match_with_concepts(query: str, choices: list[str], threshold: int = 55) -> str | None:
    if not choices or not query:
        return None
    variants     = _concept_synonyms(query)
    norm_choices = [_norm(c) for c in choices]
    best_score   = 0
    best_idx     = -1
    for variant in variants:
        if _RF:
            r = rfprocess.extractOne(variant, norm_choices, scorer=fuzz.partial_ratio)
            if r and r[1] >= threshold and r[1] > best_score:
                best_score = r[1]
                best_idx   = r[2]
        else:
            for i, nc in enumerate(norm_choices):
                if variant in nc or nc in variant:
                    return choices[i]
    return choices[best_idx] if best_idx >= 0 else None


def _pick_from_dataset(
    category:  str,
    dataset:   list[dict],
    exclude:   set[str],
    n:         int = 3,
) -> list[dict]:
    """
    Picks up to n items from dataset for the given category.
    ONLY returns items that have a valid price.
    NEVER picks items already in the shopping list.
    """
    candidates = [
        d for d in dataset
        if d.get("category", "").lower() == category.lower()
        and d["name"].lower() not in exclude
        and _has_valid_price(d)
    ]
    return candidates[:n]


def keep_only(
    category:        str,
    allowed_keywords: list[str],
    shopping_list:   list[dict],
) -> list[dict]:
    result = []
    for item in shopping_list:
        if item.get("category", "").lower() != category.lower():
            result.append(item)
            continue
        if not allowed_keywords:
            result.append(item)
            continue
        names = [item["name"]]
        kept  = (
            any(fuzzy_match(kw, names) is not None for kw in allowed_keywords) or
            any(fuzzy_match_with_concepts(kw, names) is not None for kw in allowed_keywords)
        )
        if kept:
            result.append(item)
    return result


def remove_item(keyword: str, shopping_list: list[dict]) -> list[dict]:
    if not keyword:
        return shopping_list
    names   = [item["name"] for item in shopping_list]
    matched = (
        fuzzy_match_with_concepts(keyword, names) or
        fuzzy_match(keyword, names, threshold=45)
    )
    if matched is None:
        return shopping_list
    return [item for item in shopping_list if item["name"] != matched]


def replace_category(
    category:      str,
    dataset:       list[dict],
    shopping_list: list[dict],
    n:             int = 3,
) -> list[dict]:
    kept    = [i for i in shopping_list if i.get("category", "").lower() != category.lower()]
    exclude = {i["name"].lower() for i in kept}
    new_items = _pick_from_dataset(category, dataset, exclude, n)
    return kept + new_items


def apply_modifications(
    shopping_list: list[dict],
    rules:         list[dict],
    dataset:       list[dict],
) -> list[dict]:
    result = list(shopping_list)

    for rule in rules:
        action = rule.get("action", rule.get("type", ""))

        if action == "keep_only":
            result = keep_only(
                rule.get("category", ""),
                rule.get("allowed_keywords", rule.get("allowed", [])),
                result,
            )

        elif action in ("remove", "remove_item"):
            kw = rule.get("keyword", rule.get("target", rule.get("item", "")))
            result = remove_item(kw, result)

        elif action in ("replace_category", "replace"):
            result = replace_category(
                rule.get("category", ""),
                dataset,
                result,
                rule.get("n", 3),
            )

        elif action == "increase":
            cat     = rule.get("category", "")
            exclude = {i["name"].lower() for i in result}
            extras  = _pick_from_dataset(cat, dataset, exclude, n=2)
            result  = result + extras

        elif action == "decrease":
            cat       = rule.get("category", "")
            cat_items = [i for i in result if i.get("category", "").lower() == cat.lower()]
            others    = [i for i in result if i.get("category", "").lower() != cat.lower()]
            result    = others + cat_items[:max(1, len(cat_items) - 1)]

    return result
