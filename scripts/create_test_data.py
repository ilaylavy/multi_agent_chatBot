"""
scripts/create_test_data.py — Generate realistic test data for the RAG system.

Creates:
  data/tables/employees.csv          — 10 employees, including Noa Levi (clearance A)
  data/tables/salary_bands.sqlite    — salary_bands table, 4 levels x 3 departments
  data/pdfs/travel_policy_2024.pdf   — 2-page PDF with flight and hotel policy text

Run with: python -m scripts.create_test_data
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR   = PROJECT_ROOT / "data" / "tables"
PDFS_DIR     = PROJECT_ROOT / "data" / "pdfs"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. employees.csv
# ---------------------------------------------------------------------------

EMPLOYEES = [
    # (employee_id, full_name, email, department, clearance_level, hire_date, manager_id)
    (1,  "Noa Levi",      "noa.levi@corp.com",      "Engineering", "A", "2019-03-12", ""),
    (2,  "Dan Cohen",     "dan.cohen@corp.com",      "Finance",     "B", "2018-07-01", 1),
    (3,  "Yael Ben-David","yael.bd@corp.com",        "HR",          "A", "2020-11-05", 1),
    (4,  "Oren Shapiro",  "oren.s@corp.com",         "Engineering", "C", "2021-02-20", 1),
    (5,  "Michal Katz",   "michal.k@corp.com",       "Sales",       "B", "2017-09-15", 2),
    (6,  "Tal Mizrahi",   "tal.m@corp.com",          "Finance",     "D", "2022-06-30", 2),
    (7,  "Roi Peretz",    "roi.p@corp.com",          "Engineering", "B", "2020-04-18", 1),
    (8,  "Shira Goldman", "shira.g@corp.com",        "HR",          "C", "2023-01-09", 3),
    (9,  "Amir Dayan",    "amir.d@corp.com",         "Sales",       "D", "2022-08-22", 5),
    (10, "Hila Stern",    "hila.s@corp.com",         "Engineering", "A", "2016-05-03", ""),
]

CSV_PATH = TABLES_DIR / "employees.csv"
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["employee_id", "full_name", "email", "department",
                     "clearance_level", "hire_date", "manager_id"])
    writer.writerows(EMPLOYEES)

print(f"Created {CSV_PATH}")
print(f"  Rows : {len(EMPLOYEES)}")
print(f"  Clearance breakdown: "
      + ", ".join(
          f"{lvl}={sum(1 for e in EMPLOYEES if e[4] == lvl)}"
          for lvl in ['A', 'B', 'C', 'D']
      ))
print(f"  Noa Levi: clearance={EMPLOYEES[0][4]}, department={EMPLOYEES[0][3]}")


# ---------------------------------------------------------------------------
# 2. salary_bands.sqlite
# ---------------------------------------------------------------------------

SALARY_BANDS = [
    # (id, clearance_level, department, salary_min, salary_max, currency)
    (1,  "A", "Engineering", 150000, 220000, "USD"),
    (2,  "A", "Finance",     140000, 200000, "USD"),
    (3,  "A", "HR",          130000, 185000, "USD"),
    (4,  "A", "Sales",       135000, 195000, "USD"),
    (5,  "B", "Engineering", 110000, 155000, "USD"),
    (6,  "B", "Finance",     105000, 145000, "USD"),
    (7,  "B", "HR",          95000,  135000, "USD"),
    (8,  "B", "Sales",       100000, 145000, "USD"),
    (9,  "C", "Engineering", 80000,  115000, "USD"),
    (10, "C", "Finance",     75000,  110000, "USD"),
    (11, "C", "HR",          70000,  100000, "USD"),
    (12, "C", "Sales",       75000,  110000, "USD"),
    (13, "D", "Engineering", 55000,  80000,  "USD"),
    (14, "D", "Finance",     52000,  75000,  "USD"),
    (15, "D", "HR",          50000,  72000,  "USD"),
    (16, "D", "Sales",       52000,  78000,  "USD"),
]

SQLITE_PATH = TABLES_DIR / "salary_bands.sqlite"
SQLITE_PATH.unlink(missing_ok=True)   # start fresh
conn = sqlite3.connect(str(SQLITE_PATH))
conn.execute("""
    CREATE TABLE salary_bands (
        id              INTEGER PRIMARY KEY,
        clearance_level TEXT    NOT NULL,
        department      TEXT    NOT NULL,
        salary_min      INTEGER NOT NULL,
        salary_max      INTEGER NOT NULL,
        currency        TEXT    NOT NULL DEFAULT 'USD'
    )
""")
conn.executemany(
    "INSERT INTO salary_bands VALUES (?,?,?,?,?,?)",
    SALARY_BANDS,
)
conn.commit()
row_count = conn.execute("SELECT COUNT(*) FROM salary_bands").fetchone()[0]
conn.close()

print(f"\nCreated {SQLITE_PATH}")
print(f"  Rows        : {row_count}")
print(f"  Levels      : A, B, C, D")
print(f"  Departments : Engineering, Finance, HR, Sales")
print(f"  Level A / Engineering range: $150,000 – $220,000")


# ---------------------------------------------------------------------------
# 3. travel_policy_2024.pdf
# ---------------------------------------------------------------------------

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

PDF_PATH = PDFS_DIR / "travel_policy_2024.pdf"

styles    = getSampleStyleSheet()
heading   = styles["Heading1"]
heading2  = styles["Heading2"]
body      = styles["BodyText"]
body.leading = 16

PAGE_1_CONTENT = [
    Paragraph("Travel Policy 2024", heading),
    Spacer(1, 0.4 * cm),
    Paragraph("Section 1: Flight Class Entitlements by Clearance Level", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All corporate travel must comply with the flight class entitlements defined "
        "below. Entitlements are determined solely by the employee's clearance level "
        "at the time of travel booking.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Clearance Level A:</b> "
        "Employees with clearance level A are entitled to Business Class on flights "
        "exceeding 4 hours. For flights of 4 hours or under, Economy Plus is the "
        "maximum permitted class.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level B:</b> "
        "Employees with clearance level B are entitled to Premium Economy on flights "
        "exceeding 6 hours. Economy class applies to all shorter flights.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level C:</b> "
        "Employees with clearance level C are entitled to Economy class on all flights, "
        "regardless of duration.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level D:</b> "
        "Employees with clearance level D are entitled to Economy class on all flights. "
        "Budget carrier options should be considered where available.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Paragraph("Section 2: Booking Rules", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All flights must be booked at least 14 days in advance unless a travel "
        "exception is approved by a direct manager with clearance level A or above. "
        "Last-minute bookings without approval will not be reimbursed above Economy "
        "class rates regardless of clearance level.",
        body,
    ),
]

PAGE_2_CONTENT = [
    Paragraph("Section 3: Hotel Nightly Allowances", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Hotel accommodation is reimbursed up to the nightly limits below. "
        "Receipts are required for all hotel claims. The limits apply per night "
        "including all taxes and fees.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Clearance Level A:</b> Up to $350 per night in Tier 1 cities "
        "(New York, London, Tokyo, Sydney). Up to $250 per night elsewhere.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level B:</b> Up to $250 per night in Tier 1 cities. "
        "Up to $180 per night elsewhere.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level C:</b> Up to $180 per night in Tier 1 cities. "
        "Up to $130 per night elsewhere.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level D:</b> Up to $130 per night in Tier 1 cities. "
        "Up to $100 per night elsewhere.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Paragraph("Section 4: Per Diem Meal Rates", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Meal expenses are reimbursed on a per diem basis as follows: "
        "Clearance A — $120/day; Clearance B — $90/day; "
        "Clearance C — $70/day; Clearance D — $55/day. "
        "Receipts are not required for per diem claims within these limits. "
        "Claims above the per diem rate require itemised receipts and manager approval.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Paragraph("Section 5: Receipt and Reimbursement Process", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All reimbursement claims must be submitted within 30 days of travel completion "
        "via the expense portal. Claims submitted after 30 days will not be processed "
        "without written approval from the Finance department. Original receipts or "
        "e-receipts in PDF format are required for all individual claims above $25.",
        body,
    ),
]

def _build_pdf(path: Path, page1: list, page2: list) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    from reportlab.platypus import PageBreak
    story = page1 + [PageBreak()] + page2
    doc.build(story)

_build_pdf(PDF_PATH, PAGE_1_CONTENT, PAGE_2_CONTENT)

print(f"\nCreated {PDF_PATH}")
print(f"  Pages  : 2")
print(f"  Page 1 : Flight class entitlements (Section 1 + 2)")
print(f"  Page 2 : Hotel allowances, per diem rates, reimbursement process (Sections 3-5)")
print(f"  Key sentence present: 'Employees with clearance level A are entitled to Business Class on flights exceeding 4 hours.'")

print("\nAll test data created successfully.")
