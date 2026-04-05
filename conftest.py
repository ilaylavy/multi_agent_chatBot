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
    Create data/tables/employees.csv once before the session starts and
    remove it after the full session ends. This prevents test-ordering
    fragility: the mock suite and integration tests both rely on this file,
    and it must not be deleted between them.
    """
    csv_path = _PROJECT_ROOT / "data" / "tables" / "employees.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(_EMPLOYEES_CSV_CONTENT)
    yield
    if csv_path.exists():
        csv_path.unlink()
