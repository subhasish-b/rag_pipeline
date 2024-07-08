import streamlit as st
import requests

st.title("RAG Question Answering Bot")
st.markdown("A LLM model augmented with retrieval from sentence simillarity model, to answer queries on a Concept Of Biology Book")
url = 'http://127.0.0.1:8000/rag'
query = st.text_area("Enter your query here!")
submit = st.button("Answer Me")

if submit and query:
    params = {"query" : query}
    response = requests.get(url, params= params)
    st.write(response.json())
