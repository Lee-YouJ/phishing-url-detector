"""
URL 문자열에서 모델 학습에 사용한 19개 feature를 추출합니다.
feature 이름과 순서는 학습 데이터(final_url_dataset.csv)의 숫자형 컬럼 순서와 동일합니다.
"""

import re
from urllib.parse import urlparse

# 학습 시 사용한 feature 순서 (select_features 반환 순서와 동일)
FEATURE_ORDER = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "num_digits",
    "digit_ratio",
    "num_dots",
    "num_slashes",
    "num_hyphens",
    "num_at",
    "num_question_marks",
    "num_ampersand",
    "num_special_chars",
    "subdomain_count",
    "tld_length",
    "has_punycode",
    "has_suspicious_keyword",
    "has_https",
    "has_ip",
]

_SUSPICIOUS_KEYWORDS = ["login", "verify", "update", "secure", "account"]
_IP_PATTERN = re.compile(r"(\d{1,3}\.){3}\d{1,3}")


def _normalize_url(url: str) -> str:
    """scheme이 없는 URL에 http://를 붙여 urlparse가 domain을 올바르게 파싱하도록 한다."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _canonical_url(url: str) -> str:
    """
    www. 정규화: host의 leading www. 을 제거한 canonical URL 문자열을 반환한다.
    feature 계산 시 www.naver.com 과 naver.com 이 동일한 값을 갖도록 한다.
    scheme, path, query 등 나머지는 그대로 유지한다.
    """
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path.split("/")[0]
    host = host.split("@")[-1].split(":")[0]
    canonical_host = host[4:] if host.lower().startswith("www.") else host
    if parsed.netloc:
        canonical_url = url.replace(parsed.netloc, canonical_host, 1)
    else:
        canonical_url = url
    return canonical_url, canonical_host, host


def extract_features(url: str) -> dict:
    """
    URL 문자열 하나를 받아 19개 feature를 dict로 반환합니다.
    반환 키 순서는 FEATURE_ORDER와 동일합니다.

    www 정규화: host의 leading www. 을 제거한 canonical URL 기준으로
    num_dots, url_length, domain_length 등을 계산합니다.
    - www.naver.com 과 naver.com 이 동일한 feature를 갖도록 처리합니다.
    - www 는 subdomain_count 에 포함하지 않습니다.
    """
    url = _normalize_url(url)
    parsed_orig = urlparse(url)

    # www. 정규화 적용
    canonical, canonical_host, original_host = _canonical_url(url)
    parsed = urlparse(canonical)

    domain = canonical_host
    path = parsed.path
    query = parsed.query

    tld = domain.split(".")[-1] if "." in domain else ""

    # subdomain_count: www 제거 후 canonical host 기준
    # naver.com → 0, accounts.google.com → 1, login.secure.example.com → 2
    subdomain_count = max(domain.count(".") - 1, 0)

    num_digits = sum(c.isdigit() for c in canonical)
    url_len = len(canonical)

    features = {
        "url_length":             url_len,
        "domain_length":          len(domain),
        "path_length":            len(path),
        "query_length":           len(query),
        "num_digits":             num_digits,
        "digit_ratio":            num_digits / url_len if url_len > 0 else 0.0,
        "num_dots":               canonical.count("."),
        "num_slashes":            canonical.count("/"),
        "num_hyphens":            canonical.count("-"),
        "num_at":                 canonical.count("@"),
        "num_question_marks":     canonical.count("?"),
        "num_ampersand":          canonical.count("&"),
        "num_special_chars":      len(re.findall(r"[^a-zA-Z0-9]", canonical)),
        "subdomain_count":        subdomain_count,
        "tld_length":             len(tld),
        "has_punycode":           int("xn--" in domain),
        "has_suspicious_keyword": int(any(k in path.lower() for k in _SUSPICIOUS_KEYWORDS)),
        "has_https":              int(parsed_orig.scheme == "https"),
        "has_ip":                 int(bool(_IP_PATTERN.search(domain))),
    }

    # FEATURE_ORDER 순서로 정렬된 dict 반환
    return {k: features[k] for k in FEATURE_ORDER}
