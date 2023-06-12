import os, sys
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import urllib.request as libreq
import feedparser
import pandas as pd


def sum_ab(ab):
    llm = OpenAI(temperature=0, max_tokens=2000)
    text = f'''Summarize it with 3 sentence. using Chinese:\n
    {ab}
    '''
    response = llm(text)
    return response

def search_arxiv(keyword, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{keyword}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    url = base_url + query

    with libreq.urlopen(url) as response:
        raw_response = response.read()
        
    parsed_response = feedparser.parse(raw_response)

    papers = parsed_response.entries

    results = []
    for paper in papers:
        author = ', '.join(author.name for author in paper.authors)
        result = {'Title': paper.title,
            'Summary': paper.summary,
            'Published': paper.published,
            'Authors': author,
            'Link': paper.link}
        results.append(result)
    return results

# Title
st.sidebar.title('Summarize')

# Text input
keyword = st.sidebar.text_input('Enter keyword')

# Button
button_clicked = st.sidebar.button('Search')

if button_clicked:
    results = search_arxiv(keyword)
    if results:
        for i, result in enumerate(results):
            st.sidebar.write(f'{result["Title"]}')
            st.write(f'{result["Title"]}')
            st.write(f'{result["Authors"]}')
            st.write(f'{sum_ab(result["Summary"])}')
            st.write(f'{result["Link"]}')
            st.write(f'{result["Published"]}')
            st.write('---')
            # st.sidebar.button(f'Summarize {i}')
    else:
        st.write("No results found.")
    

