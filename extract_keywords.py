"""
URL 키워드 크롤러

입력 파일(yaml/txt/json)에서 홈페이지 URL을 읽고, 각 도메인 내 모든 하위 페이지와
이미지(OCR)를 탐색하여 단어별 총 등장 횟수·고유 URL 수를 CSV로 저장합니다.

사용법:
    python extract_keyword.py <url_list_file> [output.csv]

입력 파일 형식 (권장순):
    - yaml: URL별 개별 설정·주석 지원 → urls.yaml 참고
    - txt : 줄마다 URL 하나, # 주석 가능
    - json: URL 배열 ["https://..."] 또는 {"urls": [...]}
"""

import base64
import csv
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from kiwipiepy import Kiwi
from PIL import Image
import pytesseract
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────────────────────────────────

MAX_PAGES_PER_DOMAIN = 500      # 도메인당 최대 크롤링 페이지 수
REQUEST_TIMEOUT      = 10       # HTTP 요청 타임아웃 (초)
CRAWL_DELAY          = 0.3      # 요청 간 대기 시간 (초, 과부하 방지)
OCR_LANGS            = "kor+eng+jpn+chi_sim"  # Tesseract 언어
MIN_WORD_LEN         = 2        # 단어 최소 길이 (글자 수)
JS_FALLBACK_THRESHOLD = 200     # 추출 텍스트가 이 미만이면 JS 렌더링 재시도
IMAGE_EXTS           = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# 크롤링에서 제외할 경로 세그먼트 (로그인·회원 전용 페이지)
_SKIP_SEGMENTS = frozenset({
    # 로그인/로그아웃
    "login", "logout", "signin", "signout", "sign-in", "sign-out", "log-in", "log-out",
    # 회원가입
    "register", "signup", "sign-up", "join", "membership",
    # 마이페이지/회원
    "mypage", "my-page", "my_page", "myaccount", "my-account", "my_account", "member",
    # 주문/결제
    "order", "orders", "order-history", "checkout",
    # 장바구니
    "cart",
    # 위시리스트/즐겨찾기
    "wishlist", "wish-list", "wish_list", "favorites", "favourites",
    # 계정/설정
    "account", "profile", "settings",
    # 인증
    "auth", "oauth",
    # 도움말
    "help",
})

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

_kiwi = Kiwi()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 입력 파일 파싱 ────────────────────────────────────────────────────────────

# URL 항목 타입: url 필드 필수, 나머지는 선택적 개별 설정
UrlEntry = dict  # {"url": str, "label": str, "max_pages": int, "crawl_delay": float}


def _normalize_url(u: str) -> str:
    u = u.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u


def _parse_entry(item: object) -> UrlEntry:
    """항목 하나를 UrlEntry dict로 정규화."""
    if isinstance(item, str):
        return {"url": _normalize_url(item)}
    if isinstance(item, dict):
        if "url" not in item:
            raise ValueError(f"URL 항목에 'url' 키가 없습니다: {item}")
        entry = dict(item)
        entry["url"] = _normalize_url(entry["url"])
        return entry
    raise ValueError(f"지원하지 않는 URL 항목 형식: {item!r}")


def load_urls(file_path: str) -> tuple[list[UrlEntry], dict]:
    """
    입력 파일을 읽어 (url_entries, global_settings)를 반환한다.

    global_settings 키: max_pages, crawl_delay, ocr_langs
    url_entries 키:     url, label (선택), max_pages (선택), crawl_delay (선택)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {file_path}")

    suffix = path.suffix.lower()
    global_settings: dict = {}

    # ── YAML ──
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, list):
            # 최상위가 리스트 → URL만 있는 단순 형식
            entries = [_parse_entry(item) for item in data]
        elif isinstance(data, dict):
            global_settings = data.get("settings", {})
            raw_urls = data.get("urls", [])
            if not raw_urls:
                raise ValueError("YAML 파일에 'urls' 키가 없습니다.")
            entries = [_parse_entry(item) for item in raw_urls]
        else:
            raise ValueError("YAML 파일은 URL 리스트 또는 {settings: ..., urls: [...]} 형식이어야 합니다.")

    # ── JSON ──
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            entries = [_parse_entry(item) for item in data]
        elif isinstance(data, dict):
            global_settings = data.get("settings", {})
            raw_urls = data.get("urls", [])
            entries = [_parse_entry(item) for item in raw_urls]
        else:
            raise ValueError("JSON은 URL 배열 또는 {\"urls\": [...]} 형식이어야 합니다.")

    # ── TXT (기본) ──
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        entries = [
            {"url": _normalize_url(l.strip())}
            for l in lines
            if l.strip() and not l.startswith("#")
        ]

    if not entries:
        raise ValueError("URL 항목이 하나도 없습니다.")

    return entries, global_settings

# ── HTTP 유틸리티 ─────────────────────────────────────────────────────────────

def fetch(url: str, session: requests.Session) -> requests.Response | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logger.debug("요청 실패 %s: %s", url, e)
        return None


def get_origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def is_same_origin(url: str, origin: str) -> bool:
    p = urlparse(url)
    # www 제거 후 비교하여 서브도메인 허용
    base_host = urlparse(origin).netloc.lstrip("www.")
    u_host    = p.netloc.lstrip("www.")
    return u_host == base_host or u_host.endswith("." + base_host)


def _is_member_page(url: str) -> bool:
    path = urlparse(url).path.lower()
    for seg in path.strip("/").split("/"):
        stem = seg.rsplit(".", 1)[0] if "." in seg else seg  # login.html → login
        if stem in _SKIP_SEGMENTS:
            return True
    return False

# ── 링크 수집 ─────────────────────────────────────────────────────────────────

def collect_links(soup: BeautifulSoup, page_url: str, origin: str) -> list[str]:
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full = urljoin(page_url, href)
        p = urlparse(full)
        clean = p._replace(fragment="").geturl()
        if is_same_origin(clean, origin):
            links.append(clean)
    return links

# ── 텍스트 추출 ───────────────────────────────────────────────────────────────

_SKIP_TAGS = {"script", "style", "meta", "link", "noscript", "footer"}


def html_to_text(soup: BeautifulSoup) -> str:
    for tag in soup(_SKIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=lambda c: c and "footer" in c.lower()):
        tag.decompose()
    for tag in soup.find_all(id=lambda i: i and "footer" in i.lower()):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _ocr_bytes(img_bytes: bytes) -> str:
    try:
        pil = Image.open(BytesIO(img_bytes))
        if pil.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil, mask=pil.split()[-1])
            pil = bg
        elif pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pytesseract.image_to_string(pil, lang=OCR_LANGS)
    except Exception as e:
        logger.debug("OCR 실패: %s", e)
        return ""


def images_to_text(soup: BeautifulSoup, page_url: str, session: requests.Session) -> str:
    parts = []
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if not src:
            continue

        # data URI 이미지
        if src.startswith("data:image"):
            try:
                _, b64 = src.split(",", 1)
                text = _ocr_bytes(base64.b64decode(b64))
                if text.strip():
                    parts.append(text)
            except Exception:
                pass
            continue

        img_url = urljoin(page_url, src)
        ext = Path(urlparse(img_url).path).suffix.lower()
        if ext and ext not in IMAGE_EXTS:
            continue

        resp = fetch(img_url, session)
        if resp is None or not resp.content:
            continue
        ctype = resp.headers.get("content-type", "")
        if "image" not in ctype and ext not in IMAGE_EXTS:
            continue

        text = _ocr_bytes(resp.content)
        if text.strip():
            parts.append(text)

    return " ".join(parts)

# ── 토크나이저 ────────────────────────────────────────────────────────────────

_POS_KEEP = {
    "NNG": "명사",
    "NNP": "고유명사",
    "VV":  "동사",
    "VA":  "형용사",
    "MAG": "부사",
    "SL":  "외국어",
}
_LEMMATIZE_POS = {"VV", "VA"}


_CTX_WINDOW = 40  # 단어 앞뒤로 추출할 문자 수


def tokenize(text: str) -> list[tuple[str, str, str]]:
    tokens = []
    for token in _kiwi.tokenize(text):
        pos = token.tag
        if pos not in _POS_KEEP:
            continue
        lemma = token.form
        if pos in _LEMMATIZE_POS:
            lemma += "다"
        if pos == "SL":
            lemma = lemma.lower()
        if len(lemma) < MIN_WORD_LEN:
            continue
        s = max(0, token.start - _CTX_WINDOW)
        e = min(len(text), token.start + token.len + _CTX_WINDOW)
        ctx = " ".join(text[s:e].split())
        tokens.append((lemma, _POS_KEEP[pos], ctx))
    return tokens

# ── 도메인 크롤러 ─────────────────────────────────────────────────────────────

WordData = dict[tuple[str, str], list]  # (word, pos) -> [total_count, {urls}, {brands}]
UrlLog   = list[dict]                   # [{"label": str, "url": str, "word_count": int}, ...]
FullLog  = list[tuple]                  # [(brand, url, word, pos, context), ...]


def crawl_domain(entry: UrlEntry, global_settings: dict) -> tuple[WordData, UrlLog, FullLog]:
    """BFS로 entry['url']과 같은 도메인 내 모든 페이지를 탐색."""
    start_url   = entry["url"]
    max_pages   = entry.get("max_pages")   or global_settings.get("max_pages")   or MAX_PAGES_PER_DOMAIN
    crawl_delay = entry.get("crawl_delay") or global_settings.get("crawl_delay") or CRAWL_DELAY
    label       = entry.get("label", urlparse(start_url).netloc)

    origin  = get_origin(start_url)
    visited: set[str] = set()
    queued:  set[str] = {start_url}
    queue:   list[str] = [start_url]
    data:      WordData   = defaultdict(lambda: [0, set(), set()])
    url_log:   UrlLog     = []
    full_log:  FullLog    = []
    seen_full: set[tuple] = set()
    crawled:   int        = 0  # 회원 페이지 제외한 실제 크롤링 수

    session = requests.Session()

    # Playwright 관련 상태 (필요 시 lazy 초기화)
    _pw = _browser = _pw_page = None

    def _get_js_html(target_url: str) -> str | None:
        nonlocal _pw, _browser, _pw_page
        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright 미설치 — JS 렌더링 불가 (pip install playwright && playwright install chromium)")
            return None
        try:
            if _pw_page is None:
                logger.info("  JS 렌더링 모드 활성화 (Playwright)")
                _pw      = _sync_playwright().start()
                _browser = _pw.chromium.launch(headless=True)
                _pw_page = _browser.new_page()
                _pw_page.set_extra_http_headers({"User-Agent": HEADERS["User-Agent"]})
            _pw_page.goto(target_url, wait_until="networkidle", timeout=30_000)
            return _pw_page.content()
        except Exception as e:
            logger.debug("JS 렌더링 실패 %s: %s", target_url, e)
            return None

    try:
        with tqdm(desc=f"  {label}", unit="page", dynamic_ncols=True) as bar:
            while queue and crawled < max_pages:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                if _is_member_page(url):
                    logger.debug("  회원 페이지 제외: %s", url)
                    continue

                resp = fetch(url, session)
                if resp is None:
                    url_log.append({"label": label, "url": url, "word_count": 0})
                    crawled += 1
                    bar.update(1)
                    continue

                if _is_member_page(resp.url):
                    logger.debug("  로그인 리다이렉트 제외: %s", url)
                    continue

                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype:
                    url_log.append({"label": label, "url": url, "word_count": 0})
                    crawled += 1
                    bar.update(1)
                    continue

                try:
                    soup = BeautifulSoup(resp.text, "lxml")
                except Exception:
                    url_log.append({"label": label, "url": url, "word_count": 0})
                    crawled += 1
                    bar.update(1)
                    continue

                text = html_to_text(soup)
                if len(text.strip()) < JS_FALLBACK_THRESHOLD:
                    rendered = _get_js_html(url)
                    if rendered:
                        soup = BeautifulSoup(rendered, "lxml")
                        text = html_to_text(soup)

                text += " " + images_to_text(soup, url, session)
                tokens = tokenize(text)
                for lemma, pos, ctx in tokens:
                    key = (lemma, pos)
                    data[key][0] += 1
                    data[key][1].add(url)
                    data[key][2].add(label)
                    full_key = (url, lemma, pos)
                    if full_key not in seen_full:
                        seen_full.add(full_key)
                        full_log.append((label, url, lemma, pos, ctx))
                url_log.append({"label": label, "url": url, "word_count": len(tokens)})

                for link in collect_links(soup, url, origin):
                    if link not in visited and link not in queued:
                        queued.add(link)
                        queue.append(link)

                crawled += 1
                bar.update(1)
                bar.set_postfix({"queue": len(queue), "words": len(data)})
                time.sleep(crawl_delay)
    finally:
        if _pw_page:  _pw_page.close()
        if _browser:  _browser.close()
        if _pw:       _pw.stop()

    logger.info("  완료: %d 페이지 (%d 회원페이지 제외), %d 단어", crawled, len(visited) - crawled, len(data))
    return data, url_log, full_log

# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_log(url_log: UrlLog, output_path: str, write_header: bool = False) -> None:
    mode = "w" if write_header else "a"
    with open(output_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["label", "url", "word_count"])
        writer.writerows(row.values() for row in url_log)
    logger.info("로그 업데이트: %s  (+%d개 URL)", output_path, len(url_log))


def save_full_csv(full_log: FullLog, output_path: str, write_header: bool = False) -> None:
    mode = "w" if write_header else "a"
    with open(output_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["브랜드", "url", "단어", "품사", "문맥"])
        writer.writerows(full_log)
    logger.info("전문 로그 업데이트: %s  (+%d행)", output_path, len(full_log))


def save_csv(data: WordData, output_path: str) -> None:
    rows = sorted(
        ((word, pos, d[0], len(d[1]), len(d[2])) for (word, pos), d in data.items()),
        key=lambda x: (-x[4], -x[3], -x[2]),
    )
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["단어", "품사", "총_등장_횟수", "고유_URL_수", "고유_브랜드_수"])
        writer.writerows(rows)
    logger.info("저장 완료: %s  (%d개 단어)", output_path, len(rows))

# ── 진입점 ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file  = sys.argv[1]
    raw_output  = sys.argv[2] if len(sys.argv) >= 3 else "keywords.csv"

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = str(output_dir / Path(raw_output).name)

    entries, global_settings = load_urls(input_file)
    logger.info("총 %d개 URL 처리 시작", len(entries))
    if global_settings:
        logger.info("전역 설정: %s", global_settings)

    merged:    WordData = defaultdict(lambda: [0, set(), set()])
    stem       = str(Path(output_file).with_suffix(""))
    log_file:  str = stem + "_log.csv"
    full_file: str = stem + "_full.csv"

    for idx, entry in enumerate(entries, 1):
        label = entry.get("label", entry["url"])
        logger.info("[%d/%d] %s", idx, len(entries), label)
        domain_data, url_log, full_log = crawl_domain(entry, global_settings)
        for key, (cnt, url_set, brand_set) in domain_data.items():
            merged[key][0] += cnt
            merged[key][1].update(url_set)
            merged[key][2].update(brand_set)
        save_log(url_log, log_file, write_header=(idx == 1))
        save_full_csv(full_log, full_file, write_header=(idx == 1))

    save_csv(merged, output_file)


if __name__ == "__main__":
    main()
