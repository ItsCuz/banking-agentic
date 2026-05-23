import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Banking AI Agent", layout="wide")
st.title("Banking AI Agent")

message = st.text_area(
    "Customer message",
    value="I made a bank transfer this morning, but the recipient has not received the money yet.",
    height=140,
)

if st.button("Run agent", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        response = requests.post(
            f"{API_BASE_URL}/run-agent",
            json={"message": message},
            timeout=60,
        )
        if not response.ok:
            st.error(f"API Gateway returned {response.status_code}")
            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text)
            st.stop()
        data = response.json()
        st.subheader("Final Output")
        st.write(data["final_output"])
        st.subheader("Workflow Trace")
        st.json(data["trace"])

with st.sidebar:
    st.header("Gateway")
    st.write(API_BASE_URL)
    if st.button("Health check"):
        st.json(requests.get(f"{API_BASE_URL}/health", timeout=10).json())
    if st.button("Config"):
        st.json(requests.get(f"{API_BASE_URL}/config", timeout=10).json())
