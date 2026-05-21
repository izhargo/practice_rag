import re
import time
import openpyxl
import requests
from bs4 import BeautifulSoup
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = str(ROOT / "edgar_data" / "cik_list.xlsx")
OUTPUT_DIR = ROOT / "edgar_data" / "normalized_filings"
BASE_URL = "https://www.sec.gov/Archives/"
DELAY = 0.2          # 5 requests/second — well within the 10 req/s limit
USER_AGENT = "izhargo@gmail.com"


def strip_html(content: bytes) -> bytes:
    text = content.decode("utf-8", errors="replace")
    if "<HTML>" not in text.upper():
        return content
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    plain = soup.get_text(separator="\n")
    plain = plain.replace("\u00a0", " ")
    lines = [line.rstrip() for line in plain.splitlines()]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line.strip():
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).encode("utf-8")


def normalize_text(text: str) -> str:
    """Normalize quotes, dashes, and whitespace for consistent downstream matching."""
    # Straighten curly quotes and apostrophes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Normalize em/en dashes to hyphens
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # Collapse runs of spaces/tabs (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def build_filename(coname: str, fyrmo: int, secfname: str) -> str:
    name = clean_name(coname)
    year = str(fyrmo)[:4]
    month = str(fyrmo)[4:]
    suffix = secfname.split('/')[-1]
    return f"{name}_{year}_{month}_{suffix}"


def load_rows(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        cik, coname, fyrmo, fdate, form, secfname = row
        if secfname:
            rows.append({"coname": coname, "fyrmo": fyrmo, "secfname": secfname})
    return rows


def download_filing(session: requests.Session, row: dict) -> bool:
    url = BASE_URL + row["secfname"]
    filename = build_filename(row["coname"], row["fyrmo"], row["secfname"])
    local_path = OUTPUT_DIR / filename

    if local_path.exists():
        return True  # already downloaded, skip

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        clean_bytes = strip_html(response.content)
        normalized = normalize_text(clean_bytes.decode("utf-8", errors="replace"))
        local_path.write_text(normalized, encoding="utf-8")
        return True
    except requests.HTTPError as e:
        print(f"  HTTP error {e.response.status_code}: {url}")
        return False
    except Exception as e:
        print(f"  Error: {e} — {url}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(XLSX_PATH)
    total = len(rows)
    print(f"Found {total} filings to download")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    success, failed = 0, 0
    for i, row in enumerate(rows, 1):
        ok = download_filing(session, row)
        if ok:
            success += 1
        else:
            failed += 1
        print(f"[{i}/{total}] {'OK' if ok else 'FAIL'} — {OUTPUT_DIR / build_filename(row['coname'], row['fyrmo'], row['secfname'])}")
        time.sleep(DELAY)

    print(f"\nDone. {success} downloaded, {failed} failed.")


if __name__ == "__main__":
    main()