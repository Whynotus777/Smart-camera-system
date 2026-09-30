from collections import namedtuple

import pytest

from data_ops import budget

Usage = namedtuple("Usage", "total used free")
TB = 10**12


@pytest.fixture
def disk(monkeypatch, tmp_path):
    state = {"total": 10 * TB, "free": 5 * TB}
    monkeypatch.setattr(budget.shutil, "disk_usage",
                        lambda p: Usage(state["total"], state["total"] - state["free"], state["free"]))
    monkeypatch.setenv("SCS_DATA_ROOT", str(tmp_path / "does" / "not" / "exist"))
    return state


def test_floor_and_spendable(disk):
    st = budget.disk_state()
    assert st.floor == int(10 * TB * 0.15)
    assert st.spendable == 5 * TB - int(1.5 * TB)


def test_require_allows_up_to_the_floor_and_refuses_past_it(disk):
    budget.require(int(3.5 * TB))  # leaves exactly 1.5 TB = 15%
    with pytest.raises(budget.BudgetError, match="refusing"):
        budget.require(int(3.5 * TB) + 1)


def test_already_below_floor_refuses_everything(disk):
    disk["free"] = 1 * TB
    assert budget.disk_state().spendable == 0
    with pytest.raises(budget.BudgetError):
        budget.require(1)


def test_cli_exit_codes(disk, capsys):
    assert budget.main(["--need-gb", "100"]) == 0
    assert "OK" in capsys.readouterr().out
    assert budget.main(["--need-gb", "4000"]) == 1
    assert "REFUSED" in capsys.readouterr().out
