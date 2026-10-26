"""Multipage adapter: run the existing Sentinel app without replacing it."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app_path = ROOT / "app2.py"
inference_path = ROOT / "inference2.py"
if not app_path.is_file() or not inference_path.is_file():
    import streamlit as st
    st.set_page_config(page_title="Sentinel unavailable", page_icon="🛡️", layout="wide")
    st.error("Sentinel needs the existing app2.py and inference2.py beside app3.py.")
    st.page_link("app3.py", label="← Back to Project Hub")
else:
    # Inject only a navigation link into the in-memory source after page configuration.
    # The actual app2.py file is not edited or copied; the Sentinel UI remains the source of truth.
    import streamlit as st
    source = app_path.read_text(encoding="utf-8")
    page_config = '''st.set_page_config(
    page_title="Sentinel · Hybrid Visual Inspection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
'''
    if page_config in source:
        source = source.replace(
            page_config,
            page_config + '\nst.sidebar.page_link("app3.py", label="← Project Hub", icon=":material/home:")\n',
            1,
        )
    namespace = {"__name__": "__main__", "__file__": str(app_path), "__package__": None}
    exec(compile(source, str(app_path), "exec"), namespace)
