from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rag_evaluator.application.types import DashboardLaunchSummary
from rag_evaluator.dashboard.data import DEFAULT_DATABASE_PATH


def launch_dashboard(
    *,
    database_path: Path | None = None,
) -> DashboardLaunchSummary:
    """
    Launch the Streamlit dashboard through the Python module entrypoint.
    """

    resolved_database_path = database_path or DEFAULT_DATABASE_PATH
    app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
    env = {
        **os.environ,
        "RAG_EVALUATOR_DATABASE_PATH": str(resolved_database_path),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
        env=env,
    )
    return DashboardLaunchSummary(
        database_path=resolved_database_path,
        app_path=app_path,
        return_code=completed.returncode,
    )
