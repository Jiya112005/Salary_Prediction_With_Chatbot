import requests, json
import streamlit as st
API_KEY = st.secrets["GEMINI_API_KEY"]
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=" + API_KEY

payload = {
    "contents": [
        {"parts": [{"text": "Hello Gemini"}]}
    ]
}

res = requests.post(URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
print(res.status_code, res.text)
