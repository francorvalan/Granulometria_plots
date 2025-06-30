@ECHO OFF
CD /D "%~dp0"
CALL conda activate streamlit
CALL streamlit run Streamlit.py
PAUSE