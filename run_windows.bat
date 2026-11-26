@echo off
py -m pip install -r requirements.txt
py -m streamlit run app/app.py
pause
