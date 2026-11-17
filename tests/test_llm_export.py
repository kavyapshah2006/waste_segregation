import os
import subprocess
from pathlib import Path

import pytest

from models.llm.export import MODES, build_command, bundled_config, main

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", sorted(MODES))
def test_bundled_configs_exist(mode):
    assert bundled_config(mode).exists()


def test_int4_command_adds_new_keys_with_plus(tmp_path):
    cmd = build_command("int4", Path("ckpt.pth"), Path("params.json"), tmp_path / "m.pte", 1024)
    assert cmd[1:3] == ["-m", "executorch.extension.llm.export.export_llm"]
    assert cmd[4].endswith("llama_xnnpack.yaml")
    assert "+base.model_class=llama3_2" in cmd
    assert "+base.checkpoint=ckpt.pth" in cmd
    assert "+export.max_seq_length=1024" in cmd  # not in llama_xnnpack.yaml


def test_spinquant_command_overrides_existing_keys_without_plus(tmp_path):
    cmd = build_command("spinquant", Path("c.pth"), Path("p.json"), tmp_path / "m.pte", 512)
    assert "export.max_seq_length=512" in cmd  # already in llama_xnnpack_spinquant.yaml
    assert "export.max_context_length=512" in cmd


def test_checkpoint_is_optional(tmp_path):
    cmd = build_command("int4", None, Path("p.json"), tmp_path / "m.pte")
    assert not any("base.checkpoint" in part for part in cmd)


def test_dry_run_prints_and_does_not_export(tmp_path, capsys):
    out = tmp_path / "x.pte"
    assert main(["--params", "p.json", "--output", str(out), "--dry-run"]) == 0
    assert "export_llm" in capsys.readouterr().out
    assert not out.exists()


@pytest.mark.parametrize("script", sorted((REPO / "scripts").glob("*.sh")), ids=lambda p: p.name)
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_llama_download_requires_licence_flag():
    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "download_models.sh"), "llama-spinquant"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": "/usr/bin:/bin"},  # no uvx: proves nothing is downloaded
    )
    assert result.returncode == 1
    assert "--accept-llama-license" in result.stderr
