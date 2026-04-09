"""
scripts/create_test_data.py — Generate realistic test data for the RAG system.

Creates:
  tests/fixtures/tables/employees.csv              — 10 employees, including Noa Levi (clearance A)
  tests/fixtures/tables/salary_bands.sqlite        — salary_bands table, 4 levels x 4 departments
  tests/fixtures/tables/projects.csv               — 12 projects across 4 departments
  tests/fixtures/tables/training_records.csv       — 25 training records for all 10 employees
  tests/fixtures/pdfs/travel_policy_2024.pdf       — 4-page PDF with flight, hotel, per diem, reimbursement policy
  tests/fixtures/pdfs/hr_handbook_v3.pdf           — 4-page PDF with onboarding, leave, reviews, conduct
  tests/fixtures/pdfs/it_security_policy.pdf       — 5-page PDF with passwords, data classification, incidents, access matrix, severity
  tests/fixtures/pdfs/finance_policy_2024.pdf      — 4-page PDF with expense thresholds, project budgets, reimbursements, vendor terms

All fixtures are written to tests/fixtures/ so they are committed to git.
Real (user-supplied) data goes in data/, which is gitignored.

Run with: python -m scripts.create_test_data
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
TABLES_DIR        = PROJECT_ROOT / "tests" / "fixtures" / "tables"
FIXTURE_PDFS_DIR  = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIXTURE_PDFS_DIR.mkdir(parents=True, exist_ok=True)


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
# 3. projects.csv
# ---------------------------------------------------------------------------

PROJECTS = [
    # (project_id, project_name, department, lead_employee_id, budget, spent, status, start_date, end_date)
    (1,  "Cloud Migration Phase 1",    "Engineering", 1,  120000, 115000, "Completed", "2024-01-15", "2024-06-30"),
    (2,  "Annual Audit Automation",    "Finance",     5,   45000,  48000, "Completed", "2024-02-01", "2024-05-15"),
    (3,  "Employee Portal Redesign",   "HR",          3,   60000,  35000, "Active",    "2024-06-01", ""),
    (4,  "CRM Integration",           "Sales",        5,   80000,  82500, "Completed", "2024-01-20", "2024-07-10"),
    (5,  "Cloud Migration Phase 2",    "Engineering", 10,  95000,  62000, "Active",    "2024-07-01", ""),
    (6,  "Compliance Training System", "HR",          3,   30000,  28000, "Completed", "2024-03-10", "2024-08-20"),
    (7,  "Revenue Dashboard",         "Finance",      5,   55000,  12000, "On Hold",   "2024-08-01", ""),
    (8,  "API Gateway Upgrade",       "Engineering",  7,   70000,  70500, "Completed", "2024-04-01", "2024-09-15"),
    (9,  "Lead Scoring Model",        "Sales",        5,   40000,  18000, "Active",    "2024-09-01", ""),
    (10, "Data Lake Setup",           "Engineering",  1,  150000,  45000, "On Hold",   "2024-10-01", ""),
    (11, "Benefits Platform Migration","HR",          3,   35000,  34000, "Completed", "2024-02-15", "2024-06-01"),
    (12, "Sales Forecasting Tool",    "Sales",        5,   65000,  63000, "Completed", "2024-03-01", "2024-08-30"),
]

PROJECTS_CSV_PATH = TABLES_DIR / "projects.csv"
with open(PROJECTS_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project_id", "project_name", "department", "lead_employee_id",
                     "budget", "spent", "status", "start_date", "end_date"])
    writer.writerows(PROJECTS)

active_count   = sum(1 for p in PROJECTS if p[6] == "Active")
on_hold_count  = sum(1 for p in PROJECTS if p[6] == "On Hold")
over_budget    = sum(1 for p in PROJECTS if p[5] > p[4])
print(f"\nCreated {PROJECTS_CSV_PATH}")
print(f"  Rows        : {len(PROJECTS)}")
print(f"  Active: {active_count}, On Hold: {on_hold_count}, Completed: {len(PROJECTS) - active_count - on_hold_count}")
print(f"  Over budget : {over_budget}")
print(f"  Sample rows:")
for row in PROJECTS[:3]:
    print(f"    {row}")


# ---------------------------------------------------------------------------
# 4. training_records.csv
# ---------------------------------------------------------------------------

TRAINING_RECORDS = [
    # (record_id, employee_id, training_name, completion_date, expiry_date, status, score)
    (1,  1,  "Security Awareness",      "2024-01-10", "2025-01-10", "Completed",   92),
    (2,  1,  "Data Privacy",            "2024-03-15", "2025-03-15", "Completed",   88),
    (3,  1,  "Leadership Fundamentals", "2024-06-20", "2025-06-20", "Completed",   95),
    (4,  2,  "Security Awareness",      "2023-11-05", "2024-11-05", "Expired",     78),
    (5,  2,  "Financial Compliance",    "2024-04-12", "2025-04-12", "Completed",   85),
    (6,  3,  "Security Awareness",      "2024-02-18", "2025-02-18", "Completed",   90),
    (7,  3,  "Data Privacy",            "2024-05-22", "2025-05-22", "Completed",   87),
    (8,  3,  "Leadership Fundamentals", "",           "",           "In Progress", ""),
    (9,  4,  "Security Awareness",      "2024-01-25", "2025-01-25", "Completed",   72),
    (10, 4,  "Emergency Procedures",    "2023-08-10", "2024-08-10", "Expired",     65),
    (11, 5,  "Security Awareness",      "2024-03-01", "2025-03-01", "Completed",   91),
    (12, 5,  "Financial Compliance",    "2024-07-14", "2025-07-14", "Completed",   89),
    (13, 5,  "Leadership Fundamentals", "2024-09-30", "2025-09-30", "Completed",   94),
    (14, 6,  "Security Awareness",      "2024-04-05", "2025-04-05", "Completed",   70),
    (15, 6,  "Data Privacy",            "2023-06-15", "2024-06-15", "Expired",     68),
    (16, 7,  "Security Awareness",      "2024-02-28", "2025-02-28", "Completed",   83),
    (17, 7,  "Emergency Procedures",    "2024-08-20", "2025-08-20", "Completed",   77),
    (18, 8,  "Security Awareness",      "2024-05-10", "2025-05-10", "Completed",   80),
    (19, 8,  "Data Privacy",            "",           "",           "In Progress", ""),
    (20, 8,  "Emergency Procedures",    "2024-06-01", "2025-06-01", "Completed",   75),
    (21, 9,  "Security Awareness",      "2024-03-20", "2025-03-20", "Completed",   69),
    (22, 9,  "Financial Compliance",    "2023-09-25", "2024-09-25", "Expired",     62),
    (23, 10, "Security Awareness",      "2024-01-05", "2025-01-05", "Completed",   96),
    (24, 10, "Leadership Fundamentals", "2024-04-18", "2025-04-18", "Completed",   98),
    (25, 10, "Data Privacy",            "2024-07-30", "2025-07-30", "Completed",   93),
]

TRAINING_CSV_PATH = TABLES_DIR / "training_records.csv"
with open(TRAINING_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["record_id", "employee_id", "training_name", "completion_date",
                     "expiry_date", "status", "score"])
    writer.writerows(TRAINING_RECORDS)

expired_count     = sum(1 for r in TRAINING_RECORDS if r[5] == "Expired")
in_progress_count = sum(1 for r in TRAINING_RECORDS if r[5] == "In Progress")
employees_covered = len({r[1] for r in TRAINING_RECORDS})
print(f"\nCreated {TRAINING_CSV_PATH}")
print(f"  Rows             : {len(TRAINING_RECORDS)}")
print(f"  Employees covered: {employees_covered}")
print(f"  Expired: {expired_count}, In Progress: {in_progress_count}, Completed: {len(TRAINING_RECORDS) - expired_count - in_progress_count}")
print(f"  Sample rows:")
for row in TRAINING_RECORDS[:3]:
    print(f"    {row}")


# ---------------------------------------------------------------------------
# PDF helper
# ---------------------------------------------------------------------------

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

styles    = getSampleStyleSheet()
heading   = styles["Heading1"]
heading2  = styles["Heading2"]
body      = styles["BodyText"]
body.leading = 16


def _build_pdf(path: Path, pages: list[list]) -> None:
    """Build a PDF with explicit page breaks between each page's content list."""
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    story: list = []
    for i, page in enumerate(pages):
        if i > 0:
            story.append(PageBreak())
        story.extend(page)
    doc.build(story)


def _pdf_page_count(path: Path) -> int:
    """Return the number of pages in a PDF file."""
    import fitz  # PyMuPDF
    with fitz.open(str(path)) as doc:
        return len(doc)


# ---------------------------------------------------------------------------
# 3. travel_policy_2024.pdf  (4 pages — replaces old 2-page version)
# ---------------------------------------------------------------------------

TRAVEL_PDF_PATH = FIXTURE_PDFS_DIR / "travel_policy_2024.pdf"

TRAVEL_PAGE_1 = [
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
        "<b>Clearance Level D:</b> "
        "Employees with clearance level D are entitled to Economy class only on all "
        "flights, regardless of duration. Budget carrier options should be considered "
        "where available.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level C:</b> "
        "Employees with clearance level C are entitled to Economy class on all flights. "
        "Business Class is permitted on flights exceeding 8 hours with prior manager "
        "approval.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level B:</b> "
        "Employees with clearance level B are entitled to Business Class on flights "
        "exceeding 4 hours. Economy or Economy Plus applies to shorter flights.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level A:</b> "
        "Employees with clearance level A are entitled to Business Class on all flights. "
        "First Class is permitted on flights exceeding 10 hours.",
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

TRAVEL_PAGE_2 = [
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
        "<b>Clearance Level D:</b> Up to $120 per night. Shared accommodation "
        "is encouraged for conferences and team events to reduce costs.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level C:</b> Up to $180 per night. Standard business hotels "
        "are preferred.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level B:</b> Up to $250 per night. Premium business hotels "
        "are permitted.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level A:</b> Up to $350 per night. Executive-tier accommodations "
        "are permitted including suite upgrades when available at no additional cost.",
        body,
    ),
]

TRAVEL_PAGE_3 = [
    Paragraph("Section 4: Per Diem Rates", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Meal and incidental expenses are reimbursed on a per diem basis. Per diem "
        "rates cover breakfast, lunch, dinner, and minor incidentals such as tips "
        "and local transport. Receipts are not required for claims within the per "
        "diem limit.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Clearance Level D:</b> $40 per day.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level C:</b> $60 per day.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level B:</b> $80 per day.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Clearance Level A:</b> $120 per day.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Claims above the per diem rate require itemised receipts and manager approval. "
        "Per diem rates are adjusted annually based on the Consumer Price Index and "
        "published in the January policy update.",
        body,
    ),
]

TRAVEL_PAGE_4 = [
    Paragraph("Section 5: Expense Reimbursement Process", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All reimbursement claims must be submitted within 30 days of travel completion "
        "via the corporate expense portal. Claims submitted after 30 days will not be "
        "processed without written approval from the Finance department head.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Receipt requirements:</b> Original receipts or e-receipts in PDF format "
        "are required for all individual claims exceeding $50. Claims of $50 or under "
        "may be submitted without a receipt but must include a written description of "
        "the expense.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Approval thresholds:</b> Expenses up to $500 require direct manager "
        "approval only. Expenses between $500 and $2,000 require manager approval plus "
        "department head sign-off. Expenses exceeding $2,000 require manager approval, "
        "department head sign-off, and Finance department approval.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Reimbursements are processed within 10 business days of approval. Payments are "
        "made to the employee's primary bank account on file. Disputed claims must be "
        "raised within 5 business days of reimbursement notification.",
        body,
    ),
]

_build_pdf(TRAVEL_PDF_PATH, [TRAVEL_PAGE_1, TRAVEL_PAGE_2, TRAVEL_PAGE_3, TRAVEL_PAGE_4])

travel_pages = _pdf_page_count(TRAVEL_PDF_PATH)
travel_size  = TRAVEL_PDF_PATH.stat().st_size
print(f"\nCreated {TRAVEL_PDF_PATH}")
print(f"  Pages : {travel_pages}  |  Size : {travel_size:,} bytes")
assert travel_pages == 4, f"Expected 4 pages, got {travel_pages}"


# ---------------------------------------------------------------------------
# 4. hr_handbook_v3.pdf  (4 pages)
# ---------------------------------------------------------------------------

HR_PDF_PATH = FIXTURE_PDFS_DIR / "hr_handbook_v3.pdf"

HR_PAGE_1 = [
    Paragraph("HR Handbook v3", heading),
    Spacer(1, 0.4 * cm),
    Paragraph("Chapter 1: Onboarding", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All new employees must complete the onboarding checklist within their first "
        "week of employment. The checklist is managed by the HR department and tracked "
        "in the onboarding portal.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>First Week Checklist:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "1. Complete and submit all required employment documents including tax forms, "
        "emergency contact information, and bank details for payroll. "
        "2. Attend the mandatory orientation session covering company values, "
        "organisational structure, and key policies. "
        "3. Collect employee ID badge and building access card from the Security desk. "
        "4. Complete IT setup: laptop configuration, email activation, VPN credentials, "
        "and access to required internal systems (HR portal, expense portal, project "
        "management tools). "
        "5. Meet with direct manager to review role expectations, 30/60/90 day goals, "
        "and team introduction schedule. "
        "6. Complete mandatory compliance training modules: data privacy, anti-harassment, "
        "and workplace safety.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Required Documents:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Government-issued photo ID, proof of right to work, signed employment contract, "
        "tax withholding form, signed NDA, emergency contact form, and bank account "
        "details for direct deposit.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>IT Setup:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "IT will provision a company laptop within 2 business days of the start date. "
        "The employee must set up multi-factor authentication on day one. Default "
        "password must be changed immediately upon first login. Access to restricted "
        "systems is granted after clearance level assignment, typically within 5 "
        "business days.",
        body,
    ),
]

HR_PAGE_2 = [
    Paragraph("Chapter 2: Leave Policy", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Leave entitlements are determined by the employee's clearance level. All leave "
        "must be requested through the HR portal at least 5 business days in advance, "
        "except for sick leave which may be reported on the day.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Annual Leave:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Clearance Level D: 15 days per year. "
        "Clearance Level C: 18 days per year. "
        "Clearance Level B: 22 days per year. "
        "Clearance Level A: 25 days per year. "
        "Unused annual leave may be carried over up to a maximum of 5 days into the "
        "following calendar year. Any leave beyond the carry-over limit is forfeited "
        "on January 1st.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Sick Leave:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "All employees receive 10 sick days per year regardless of clearance level. "
        "A medical certificate is required for absences of 3 or more consecutive days. "
        "Unused sick leave does not carry over and is not compensated.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Parental Leave:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "All employees are entitled to 90 days of paid parental leave for the birth "
        "or adoption of a child. Parental leave may be taken in a single block or split "
        "into two periods within the first 12 months. An additional 30 days of unpaid "
        "parental leave is available upon request.",
        body,
    ),
]

HR_PAGE_3 = [
    Paragraph("Chapter 3: Performance Reviews", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Performance reviews are conducted every 6 months, in June and December. "
        "Reviews are a structured conversation between the employee and their direct "
        "manager, documented in the HR portal.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Rating Scale:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Employees are rated on a scale of 1 to 5: "
        "1 = Unsatisfactory, 2 = Needs Improvement, 3 = Meets Expectations, "
        "4 = Exceeds Expectations, 5 = Outstanding. "
        "Ratings are based on goal achievement, competency demonstration, and "
        "alignment with company values.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Performance Improvement Plan (PIP):</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Any employee receiving a rating below 2 in a review cycle will be placed on "
        "a Performance Improvement Plan. The PIP lasts 90 days and includes specific, "
        "measurable goals agreed upon by the employee and manager. Progress is reviewed "
        "at 30-day intervals. Failure to meet PIP goals may result in reassignment, "
        "demotion, or termination in accordance with local employment law.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Employees who receive a rating of 4 or 5 are eligible for accelerated "
        "promotion review, salary band adjustment, and participation in the leadership "
        "development programme.",
        body,
    ),
]

HR_PAGE_4 = [
    Paragraph("Chapter 4: Code of Conduct", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Confidentiality:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "All employees are bound by the Non-Disclosure Agreement signed during onboarding. "
        "Company information classified as Confidential or Restricted must not be shared "
        "with external parties without written authorisation from the Legal department. "
        "Discussions of confidential matters in public spaces, including cafeterias and "
        "public transport, are prohibited.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Conflict of Interest:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Employees must disclose any financial interest, secondary employment, or "
        "personal relationship that could constitute a conflict of interest. Disclosures "
        "must be made to the employee's direct manager and the Ethics Committee within "
        "10 business days of the conflict arising. Failure to disclose a conflict of "
        "interest is a disciplinary offence.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Disciplinary Steps:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "The disciplinary process follows four stages: "
        "1. Verbal warning documented in the HR portal. "
        "2. Written warning with specific corrective actions required. "
        "3. Final written warning with a defined remediation period. "
        "4. Termination, subject to review by HR and Legal. "
        "Gross misconduct (fraud, harassment, theft, data breach) may result in "
        "immediate termination without prior warnings.",
        body,
    ),
]

_build_pdf(HR_PDF_PATH, [HR_PAGE_1, HR_PAGE_2, HR_PAGE_3, HR_PAGE_4])

hr_pages = _pdf_page_count(HR_PDF_PATH)
hr_size  = HR_PDF_PATH.stat().st_size
print(f"\nCreated {HR_PDF_PATH}")
print(f"  Pages : {hr_pages}  |  Size : {hr_size:,} bytes")
assert hr_pages == 4, f"Expected 4 pages, got {hr_pages}"


# ---------------------------------------------------------------------------
# 5. it_security_policy.pdf  (3 pages)
# ---------------------------------------------------------------------------

IT_PDF_PATH = FIXTURE_PDFS_DIR / "it_security_policy.pdf"

IT_PAGE_1 = [
    Paragraph("IT Security Policy", heading),
    Spacer(1, 0.4 * cm),
    Paragraph("Section 1: Password Policy", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All employees must adhere to the following password requirements for every "
        "system that supports password authentication. These requirements apply to "
        "corporate email, VPN, internal applications, and cloud services.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Minimum Length:</b> All passwords must be at least 12 characters long. "
        "Passwords shorter than 12 characters will be rejected by the system at "
        "creation or change time.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Complexity:</b> Passwords must contain at least one uppercase letter, "
        "one lowercase letter, one digit, and one special character from the set "
        "!@#$%^&amp;*()-_=+. Dictionary words and common patterns (e.g. 'Password123!') "
        "are detected and rejected.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Rotation:</b> Passwords must be changed every 90 days. The system will "
        "prompt the user 7 days before expiration. After expiration, the account is "
        "locked until a new password is set via the self-service portal or IT support.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>History:</b> The last 10 passwords are stored in hashed form. Users may "
        "not reuse any of their last 10 passwords. This prevents trivial password "
        "cycling.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Multi-Factor Authentication (MFA):</b> MFA is required for all corporate "
        "systems without exception. Supported MFA methods include hardware security "
        "keys (preferred), authenticator apps, and SMS codes (emergency fallback only). "
        "MFA must be enrolled within 24 hours of account creation.",
        body,
    ),
]

IT_PAGE_2 = [
    Paragraph("Section 2: Data Classification", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All company data must be classified into one of the following four categories. "
        "The classification determines storage, access, sharing, and retention "
        "requirements.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Public:</b> Information approved for external distribution. Examples: "
        "marketing materials, published financial reports, job postings. No access "
        "restrictions.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Internal:</b> Information intended for all employees but not for external "
        "parties. Examples: internal newsletters, org charts, general process documents. "
        "Requires company network or VPN access.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Confidential:</b> Sensitive business information with limited distribution. "
        "Examples: financial projections, strategic plans, customer contracts, unreleased "
        "product specifications. Access restricted to employees with a need-to-know "
        "basis as determined by the data owner.",
        body,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Restricted:</b> The highest classification level. Requires clearance level B "
        "or above to access. Examples: employee personal data (salaries, health records, "
        "disciplinary history), security credentials, encryption keys, legal "
        "correspondence marked privileged. Personal employee data is always classified "
        "as Restricted regardless of content.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Restricted data must be stored only on encrypted, company-managed systems. "
        "Transmission of Restricted data outside the corporate network requires "
        "end-to-end encryption and prior approval from the IT Security team. Any "
        "unauthorised access to Restricted data must be reported immediately as a "
        "security incident.",
        body,
    ),
]

IT_PAGE_3 = [
    Paragraph("Section 3: Security Incident Reporting", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "A security incident is any event that compromises, or has the potential to "
        "compromise, the confidentiality, integrity, or availability of company systems "
        "or data. Examples include unauthorised access, malware infections, phishing "
        "attacks, lost or stolen devices, and data leaks.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Reporting Deadline:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Any suspected or confirmed security incident must be reported to the IT "
        "Security team within 2 hours of discovery. Delayed reporting increases the "
        "risk of damage and may result in disciplinary action.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>How to Report:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Contact the IT Security team via the dedicated incident hotline (ext. 5555), "
        "the incident reporting form in the IT portal, or by emailing "
        "security-incident@corp.com. For critical incidents outside business hours, "
        "use the 24/7 emergency hotline listed on the IT portal homepage.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Immediate Actions:</b>", body),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "1. Preserve all relevant logs, screenshots, and evidence. Do not delete, "
        "modify, or restart affected systems. "
        "2. Disconnect the affected device from the network if instructed by IT Security. "
        "3. Do not attempt self-remediation. Well-intentioned fixes can destroy forensic "
        "evidence and complicate the response. "
        "4. Do not discuss the incident with anyone outside the response team until "
        "authorised by the IT Security lead.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "The IT Security team will acknowledge the report within 30 minutes and assign "
        "an incident severity level (Critical, High, Medium, Low). A post-incident "
        "review is conducted within 5 business days of resolution for all High and "
        "Critical incidents.",
        body,
    ),
]

IT_PAGE_4 = [
    Paragraph("Section 4: Data Access Matrix", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "The following matrix defines data access permissions by clearance level. "
        "Access is enforced at the system level and audited quarterly.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Table(
        [
            ["Data Category",         "Clearance A",  "Clearance B",  "Clearance C",  "Clearance D"],
            ["Employee Personal Data", "Full Access",  "Read Only",    "No Access",    "No Access"],
            ["Financial Records",      "Full Access",  "Full Access",  "Read Only",    "No Access"],
            ["Customer Data",          "Full Access",  "Full Access",  "Full Access",  "Read Only"],
            ["Internal Documents",     "Full Access",  "Full Access",  "Full Access",  "Full Access"],
            ["Public Data",            "Full Access",  "Full Access",  "Full Access",  "Full Access"],
        ],
        colWidths=[3.5 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#D9E2F3")]),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]),
    ),
    Spacer(1, 0.4 * cm),
    Paragraph(
        "Access requests outside of the defined matrix must be submitted to the "
        "IT Security team with justification and approved by the data owner. "
        "Temporary elevated access expires after 30 days unless renewed.",
        body,
    ),
]

IT_PAGE_5 = [
    Paragraph("Section 5: Incident Severity Classification", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All security incidents are classified by severity level. The severity "
        "determines the response time and escalation path.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Table(
        [
            ["Severity",  "Description",           "Response Time", "Escalation"],
            ["Critical",  "System-wide breach",    "30 minutes",    "CISO + Legal"],
            ["High",      "Department breach",      "2 hours",       "IT Security Lead"],
            ["Medium",    "Single user compromise", "24 hours",      "IT Security team"],
            ["Low",       "Failed access attempt",  "72 hours",      "IT Security team"],
        ],
        colWidths=[2.5 * cm, 4.5 * cm, 3.0 * cm, 4.0 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C00000")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FCE4EC")]),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]),
    ),
    Spacer(1, 0.4 * cm),
    Paragraph(
        "Critical and High incidents trigger automatic notification to the executive "
        "team. All incidents require a written incident report within 48 hours of "
        "resolution.",
        body,
    ),
]

_build_pdf(IT_PDF_PATH, [IT_PAGE_1, IT_PAGE_2, IT_PAGE_3, IT_PAGE_4, IT_PAGE_5])

it_pages = _pdf_page_count(IT_PDF_PATH)
it_size  = IT_PDF_PATH.stat().st_size
print(f"\nCreated {IT_PDF_PATH}")
print(f"  Pages : {it_pages}  |  Size : {it_size:,} bytes")
assert it_pages == 5, f"Expected 5 pages, got {it_pages}"


# ---------------------------------------------------------------------------
# 6. finance_policy_2024.pdf  (4 pages)
# ---------------------------------------------------------------------------

FINANCE_PDF_PATH = FIXTURE_PDFS_DIR / "finance_policy_2024.pdf"

FINANCE_PAGE_1 = [
    Paragraph("Finance Policy 2024", heading),
    Spacer(1, 0.4 * cm),
    Paragraph("Section 1: Expense Approval Thresholds", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All employee expenses must follow the approval thresholds defined below. "
        "The required approvers depend on the expense amount. Receipts and turnaround "
        "times are as specified.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Table(
        [
            ["Amount Range",  "Required Approvers",                   "Receipt Required", "Turnaround Time"],
            ["Up to $100",    "Self-approval",                        "No",               "Immediate"],
            ["$100 - $500",   "Manager approval",                     "Yes",              "3 business days"],
            ["$500 - $2,000", "Manager + Department Head",            "Yes",              "5 business days"],
            ["Over $2,000",   "Manager + Department Head + Finance",  "Yes",              "10 business days"],
        ],
        colWidths=[3.0 * cm, 5.5 * cm, 3.0 * cm, 3.5 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E8F5E9")]),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]),
    ),
    Spacer(1, 0.4 * cm),
    Paragraph(
        "Expenses submitted without the required approvals will be returned to the "
        "employee and must be resubmitted through the correct approval chain.",
        body,
    ),
]

FINANCE_PAGE_2 = [
    Paragraph("Section 2: Project Budget Limits by Department and Clearance Level", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "Each employee's monthly project budget limit is determined by their department "
        "and clearance level. Budgets may not be pooled or transferred between employees "
        "without Finance approval.",
        body,
    ),
    Spacer(1, 0.4 * cm),
    Table(
        [
            ["Department",   "Clearance A", "Clearance B", "Clearance C", "Clearance D"],
            ["Engineering",  "$50,000",     "$30,000",     "$15,000",     "$5,000"],
            ["Finance",      "$40,000",     "$25,000",     "$10,000",     "$3,000"],
            ["HR",           "$20,000",     "$12,000",     "$6,000",      "$2,000"],
            ["Sales",        "$45,000",     "$28,000",     "$14,000",     "$4,000"],
        ],
        colWidths=[3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#BBDEFB")]),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]),
    ),
    Spacer(1, 0.4 * cm),
    Paragraph(
        "Budget overruns exceeding 10% require written justification and approval from "
        "the department head. Overruns exceeding 25% require additional approval from "
        "the Finance department.",
        body,
    ),
]

FINANCE_PAGE_3 = [
    Paragraph("Section 3: Reimbursement Procedures", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All reimbursement claims must be submitted through the corporate expense portal "
        "within 30 days of the expense being incurred. Late submissions require written "
        "justification from the employee's manager.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Cross-References:</b> For travel expense limits, refer to Travel Policy 2024 "
        "Section 3. For hotel nightly allowances, refer to Travel Policy 2024 Section 3. "
        "For flight class entitlements, refer to Travel Policy 2024 Section 1.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Processing Time:</b> Reimbursements are processed within 10 business days "
        "of approved submission. Payments are deposited directly into the employee's "
        "primary bank account on file.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>International Expenses:</b> All international expenses require pre-approval "
        "from the Finance department regardless of amount. Currency conversion is applied "
        "at the exchange rate on the date of the transaction. Employees must provide "
        "original receipts in the local currency.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Disputed Claims:</b> Employees may dispute a rejected or partially approved "
        "reimbursement by submitting a formal appeal through the expense portal within "
        "5 business days of the rejection notification.",
        body,
    ),
]

FINANCE_PAGE_4 = [
    Paragraph("Section 4: Vendor Payment Terms", heading2),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All vendor invoices are processed according to the payment terms agreed in the "
        "vendor contract. Standard and preferred vendor terms are defined below.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Standard Vendors:</b> Net 30 payment terms. Invoices are due within 30 "
        "calendar days of receipt. Early payment discounts offered by vendors may be "
        "accepted if the discount exceeds 2% and the invoice amount exceeds $1,000.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Preferred Vendors:</b> Net 15 payment terms. Preferred vendor status is "
        "granted to vendors with a minimum 2-year relationship and annual spend exceeding "
        "$50,000. Preferred vendors receive priority invoice processing.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Small Purchases:</b> Amounts under $50 are paid immediately upon receipt "
        "of goods or services. No purchase order is required for amounts under $50.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "<b>Late Payment Penalty:</b> A penalty of 1.5% per month is applied to any "
        "invoice not paid within the agreed payment terms. The penalty is calculated on "
        "the outstanding balance from the due date until full payment is received.",
        body,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph(
        "All vendor disputes must be escalated to the Finance department within 10 "
        "business days of the disputed invoice date.",
        body,
    ),
]

_build_pdf(FINANCE_PDF_PATH, [FINANCE_PAGE_1, FINANCE_PAGE_2, FINANCE_PAGE_3, FINANCE_PAGE_4])

finance_pages = _pdf_page_count(FINANCE_PDF_PATH)
finance_size  = FINANCE_PDF_PATH.stat().st_size
print(f"\nCreated {FINANCE_PDF_PATH}")
print(f"  Pages : {finance_pages}  |  Size : {finance_size:,} bytes")
assert finance_pages == 4, f"Expected 4 pages, got {finance_pages}"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\nAll test data created successfully.")
print(f"  travel_policy_2024.pdf   : {travel_pages} pages, {travel_size:,} bytes")
print(f"  hr_handbook_v3.pdf       : {hr_pages} pages, {hr_size:,} bytes")
print(f"  it_security_policy.pdf   : {it_pages} pages, {it_size:,} bytes")
print(f"  finance_policy_2024.pdf  : {finance_pages} pages, {finance_size:,} bytes")
