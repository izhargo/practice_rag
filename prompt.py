FILING_ANALYSIS_PROMPT = """\
You are analyzing an SEC EDGAR filing. Produce a structured analysis with three parts:
a filing summary, a document title, and a list of sections with their summaries.

━━━ 1. FILING SUMMARY ━━━
Read the entire filing and write a summary of 5-6 sentences. Start with the company's core
business and industry. Then provide a concise summary of the major issues mentioned in the
report — key financial results, significant events, risks, legal matters, strategic changes,
or any other notable disclosures.

━━━ 2. SECTIONS ━━━
Divide the filing into its major sections. For each section provide the heading, a summary,
and the character positions where it starts and ends in the original text.

If the document has a TABLE OF CONTENTS, use it — its entries define the sections and their
headings. This is the most reliable source.

If there is no table of contents, here are some recommendations for identifying sections:
- Look for the standard SEC PART / ITEM structure: "PART I", "PART II", "ITEM 1", "ITEM 2",
  etc. These may appear with colons, dashes, or descriptions, and may be split across lines
  in HTML-stripped text.
- Short all-caps lines (<= 80 chars) surrounded by blank lines often serve as section headings.
- Use your judgment — these are guidelines, not strict rules.

For each section provide:
- heading: a descriptive section name. You may combine hierarchy levels for clarity,
  e.g. "Part I Financial Information Item 1 Financial Statements".
- summary: 3-4 sentence summary of what this section covers
- start_text: the EXACT first 75 characters of the section body (after the heading line).
  Copy this AS-IS from the filing text — do not rephrase, reformat, or clean up whitespace.
  This string will be used for an exact text search, so even a single changed character will
  cause a failure.
- end_text: the EXACT last 75 characters of the section body (before the next section begins).
  Same rule: copy AS-IS from the filing text, character for character.

━━━ 3. TITLE ━━━
Build a human-readable title for this filing that includes:
- Company name (use proper title case, e.g. "Sunbeam Corp" not "SUNBEAM CORP")
- Report type (e.g. "Annual Report", "Quarterly Report")
- Form type in parentheses (e.g. "Form 10-K", "Form 10-Q")
- Full reporting period with start and end dates in YYYY-MM-DD format

Recommended approach for finding these details:
- Company name: look for COMPANY CONFORMED NAME in the SGML header at the top of the file.
- Form type: look for CONFORMED SUBMISSION TYPE in the SGML header.
- Period end date: look for CONFORMED PERIOD OF REPORT in the SGML header (YYYYMMDD format).
- Period start date: this is NOT in the header — try to infer it from the filing body. Annual
  reports (10-K) typically cover ~12 months, quarterly reports (10-Q) ~3 months. The filing
  often states the period explicitly, e.g. "for the fiscal year ended ..." or "for the quarter
  ended ...". You might find such phrases near the top of the document or in the financial
  statements header. Use your best judgment if the exact start date is unclear.

Example titles:
- "Sunbeam Corp — Annual Report (Form 10-K), for the period of 1997-01-01 to 1997-12-28"
- "AT&T Corp — Quarterly Report (Form 10-Q), for the period of 1999-04-01 to 1999-06-30"

━━━ OUTPUT FORMAT ━━━
Return ONLY valid JSON — no explanation, no markdown fences:

{{
  "summary": "5-6 sentence filing summary.",
  "title": "Company Name — Report Type (Form XX), for the period of YYYY-MM-DD to YYYY-MM-DD",
  "sections": [
    {{
      "heading": "Section Name",
      "summary": "3-4 sentence section summary.",
      "start_text": "exact first 100 chars of section body",
      "end_text": "exact last 100 chars of section body"
    }}
  ]
}}

If the document has no discernible section structure (e.g. NT filings), return an empty
array for sections.

━━━ FILING TEXT ━━━
{text}"""
