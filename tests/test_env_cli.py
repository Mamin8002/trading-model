"""Тесты работы с секретами (.env, env-переменные, маскирование) и CLI validate_data."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tradebot.config.env import (
    SecretMasker,
    SecretNotFound,
    load_dotenv_file,
    resolve_optional,
    resolve_secret,
)


class TestDotEnv:
    def test_loads_key_values(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# комментарий\n"
            "FAKE_TEST_VAR_1=value1\n"
            'FAKE_TEST_VAR_2="quoted value"\n'
            "FAKE_TEST_VAR_3='single'\n"
            "\n"
            "broken-line-without-equals\n",
            encoding="utf-8",
        )
        for name in ("FAKE_TEST_VAR_1", "FAKE_TEST_VAR_2", "FAKE_TEST_VAR_3"):
            monkeypatch.delenv(name, raising=False)
        loaded = load_dotenv_file(env_file)
        assert loaded == 3
        assert os.environ["FAKE_TEST_VAR_1"] == "value1"
        assert os.environ["FAKE_TEST_VAR_2"] == "quoted value"
        assert os.environ["FAKE_TEST_VAR_3"] == "single"

    def test_existing_env_wins(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FAKE_TEST_VAR_X=from_file\n", encoding="utf-8")
        monkeypatch.setenv("FAKE_TEST_VAR_X", "from_env")
        load_dotenv_file(env_file)
        assert os.environ["FAKE_TEST_VAR_X"] == "from_env"

    def test_missing_file_is_ok(self, tmp_path):
        assert load_dotenv_file(tmp_path / "no.env") == 0


class TestResolveSecret:
    def test_missing_secret_error_message_has_no_value(self, monkeypatch):
        monkeypatch.delenv("FAKE_MISSING_SECRET", raising=False)
        with pytest.raises(SecretNotFound) as exc:
            resolve_secret("FAKE_MISSING_SECRET")
        # сообщение содержит только имя переменной и инструкцию
        assert "FAKE_MISSING_SECRET" in str(exc.value)
        assert "REPLACE_WITH" not in str(exc.value)

    def test_resolved_secret_registered_for_masking(self, monkeypatch):
        monkeypatch.setenv("FAKE_PRESENT_SECRET", "s3cr3t-fake-value")
        masker = SecretMasker()
        value = resolve_secret("FAKE_PRESENT_SECRET", masker=masker)
        assert value == "s3cr3t-fake-value"
        assert masker.mask("leak s3cr3t-fake-value here") == "leak ***SECRET*** here"

    def test_resolve_optional(self, monkeypatch):
        # локальный маскировщик: не загрязняем GLOBAL_MASKER между тестами
        masker = SecretMasker()
        monkeypatch.delenv("FAKE_OPT", raising=False)
        assert resolve_optional("FAKE_OPT", masker=masker) is None
        monkeypatch.setenv("FAKE_OPT", "fake-optional-value")
        assert resolve_optional("FAKE_OPT", masker=masker) == "fake-optional-value"
        assert masker.mask("leak fake-optional-value") == "leak ***SECRET***"


class TestCliValidateData:
    def test_demo_config_passes(self):
        from tradebot.cli.validate_data import main

        assert main(["--config", "configs/demo.yaml"]) == 0

    def test_demo_with_resample_passes(self):
        from tradebot.cli.validate_data import main

        assert main(["--config", "configs/demo.yaml", "--resample", "10m"]) == 0

    def test_single_csv_requires_semantics(self):
        from tradebot.cli.validate_data import main

        with pytest.raises(SystemExit) as exc:
            main(["--csv", "data/demo/DEMO_M1.csv", "--timeframe", "1m"])
        assert exc.value.code == 2  # argparse error

    def test_single_csv_ok(self):
        from tradebot.cli.validate_data import main

        code = main(
            [
                "--csv",
                "data/demo/DEMO_M1.csv",
                "--timeframe",
                "1m",
                "--timestamp-semantics",
                "bar_open",
            ]
        )
        assert code == 0

    def test_bad_csv_returns_1(self, tmp_path):
        from tradebot.cli.validate_data import main

        bad = tmp_path / "bad.csv"
        bad.write_text(
            "datetime,open,high,low,close,volume\n"
            "2024-01-04 10:00:00,100.0,99.0,101.0,100.5,10\n",  # high < low
            encoding="utf-8",
        )
        code = main(
            ["--csv", str(bad), "--timeframe", "1m", "--timestamp-semantics", "bar_open"]
        )
        assert code == 1

    def test_module_runnable_via_python_m(self):
        proc = subprocess.run(
            [sys.executable, "-m", "tradebot.cli.validate_data", "--config", "configs/demo.yaml"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        assert "ИТОГ: OK" in proc.stdout


class TestRepoHygiene:
    """Гигиена репозитория: секреты и торговые операции."""

    REPO_FILES_GLOBS = [
        "src/**/*.py",
        "tests/*.py",
        "scripts/**/*.py",
        "configs/*.yaml",
        ".env.example",
        ".gitignore",
        "pyproject.toml",
    ]

    def _iter_repo_files(self):
        for pattern in self.REPO_FILES_GLOBS:
            yield from Path(".").glob(pattern)

    def test_no_order_endpoints_in_code(self):
        # торговые операции не реализованы: соответствующих путей/имён в коде нет.
        # Токены собраны из фрагментов, чтобы этот тест не обнаруживал сам себя.
        forbidden = [
            "/or" + "ders",
            "Place" + "Order",
            "Cancel" + "Order",
            "sltp-" + "orders",
            "new_" + "order",
        ]
        offenders = []
        for path in self._iter_repo_files():
            if path.name in {"test_env_cli.py"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path}: {token}")
        assert offenders == [], f"Найдены упоминания торговых операций: {offenders}"

    def test_env_example_has_only_placeholders(self):
        text = Path(".env.example").read_text(encoding="utf-8")
        assert "REPLACE_WITH_YOUR_OWN" in text
        # никаких значений, похожих на JWT/ключи
        assert "eyJ" not in text

    def test_gitignore_covers_env(self):
        text = Path(".gitignore").read_text(encoding="utf-8")
        assert ".env" in text
        assert "!.env.example" in text

    def test_real_env_secret_not_in_sources(self):
        real = os.environ.get("TRADE_API_SECRET", "").strip()
        if not real:
            pytest.skip("TRADE_API_SECRET не задан в окружении — проверять нечего")
        for path in self._iter_repo_files():
            assert real not in path.read_text(encoding="utf-8", errors="replace"), path


class TestManualScript:
    def test_manual_check_script_passes(self):
        proc = subprocess.run(
            [sys.executable, "scripts/manual_checks/check_phase_02_data.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "RESULT: PASS" in proc.stdout
