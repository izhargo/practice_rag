import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

FORM_TYPE_MAP = {
    "10-K":    "Annual Report",
    "10-K405": "Annual Report",
    "10-K/A":  "Amended Annual Report",
    "10-Q":    "Quarterly Report",
    "10-Q/A":  "Amended Quarterly Report",
    "NT 10-K": "Late Filing Notification (Annual)",
    "NT 10-Q": "Late Filing Notification (Quarterly)",
}


def call_llm_using_openai_client(
    prompt: str,
    api_key: str,
    base_url: str,
    model_name: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n", "", text)
    return text.strip()


def chunk(file_name: str, chunk_size: int = 1000, overlap_size: int = 200) -> list[str]:
    with open(file_name, encoding="utf-8") as f:
        text = f.read()
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )
    return splitter.split_text(text)


def extract_title(file_path: Path) -> dict:
    """Extract metadata from the SGML header of an SEC EDGAR filing."""
    with open(file_path, encoding="utf-8", errors="replace") as f:
        words = []
        header_lines = []
        for line in f:
            header_lines.append(line)
            words.extend(line.split())
            if len(words) >= 100:
                break

    header = "".join(header_lines)

    company_match = re.search(r"COMPANY CONFORMED NAME:\s*(.+?)(?:\s*\n)", header)
    form_match = re.search(r"CONFORMED SUBMISSION TYPE:\s*(.+?)(?:\s*\n)", header)
    period_match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d+)", header)

    company_name = company_match.group(1).strip() if company_match else "UNKNOWN"
    company_name = re.sub(r"\s*/[A-Z]+/", "", company_name).strip()
    form_type = form_match.group(1).strip() if form_match else "UNKNOWN"
    period = period_match.group(1).strip() if period_match else None
    report_category = FORM_TYPE_MAP.get(form_type, "Other")
    period_label = f"{period[:4]}-{period[4:6]}-{period[6:]}" if period else ""
    title = f"{company_name} — {report_category} (Form {form_type}), period ending {period_label}" if period else f"{company_name} — {report_category} (Form {form_type})"

    return {
        "company_name": company_name,
        "form_type": form_type,
        "report_category": report_category,
        "period": period,
        "title": title,
    }