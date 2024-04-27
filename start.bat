@echo off
cd /d %~dp0

call conda activate trainer 
streamlit run app.py