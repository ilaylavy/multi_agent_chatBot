from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent

_EMPLOYEES_CSV_CONTENT = (
    "employee_id,full_name,email,department,clearance_level,hire_date,manager_id\n"
    "1,Noa Levi,noa@corp.com,Engineering,A,2020-01-15,\n"
    "2,Dan Cohen,dan@corp.com,Finance,B,2019-06-01,1\n"
    "3,Yael Ben,yael@corp.com,HR,A,2021-03-20,1\n"
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires real OpenAI API key and test data",
    )


@pytest.fixture(scope="session", autouse=True)
def employees_csv():
    """
    Ensure tests/fixtures/tables/employees.csv exists before the session starts.
    The file is a committed test fixture used by both pytest and live API sessions,
    so it is never deleted. If it already exists on disk, creation is skipped.
    """
    csv_path = _PROJECT_ROOT / "tests" / "fixtures" / "tables" / "employees.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(_EMPLOYEES_CSV_CONTENT)
    yield
