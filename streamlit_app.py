"""
streamlit_app.py

Entry point for Streamlit Community Cloud, which requires a root-level file
and clones the repo directly rather than building the Docker image -- it has
no notion of `docker-compose`'s two-container split.

The actual dashboard lives in app/app.py, run identically by
`streamlit run app/app.py` in docker-compose.yml. This just runs that same
file as the page, so there is exactly one implementation to keep correct
regardless of which of the two ways it gets deployed.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app" / "app.py"), run_name="__main__")
