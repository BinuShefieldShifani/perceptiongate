from pathlib import Path

from typer.testing import CliRunner

from perceptiongate import __version__
from perceptiongate.cli import app

runner = CliRunner()
BASELINE = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "baseline.yaml"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_validate_config_ok() -> None:
    result = runner.invoke(app, ["validate-config", str(BASELINE)])
    assert result.exit_code == 0
    assert "OK  baseline" in result.stdout
