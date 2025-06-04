@ECHO OFF
CD /D "%~dp0"
CALL conda activate sam
CALL streamlit run Streamlit.py
PAUSE