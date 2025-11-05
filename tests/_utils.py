import shutil
import subprocess
import pytest
import shlex

def check_cli_or_skip(cli: list[str]) -> None:
    """Skip the test if the CLI or subcommand isn't available."""
    # launcher on PATH?
    exe = cli[0]
    if shutil.which(exe) is None:
        pytest.skip(f"'{exe}' not found on PATH")

    # subcommand exists?
    probe = subprocess.run(
        cli + ["--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        cmd = shlex.join(cli)
        pytest.skip(f"Subcommand '{cmd}' not available")

