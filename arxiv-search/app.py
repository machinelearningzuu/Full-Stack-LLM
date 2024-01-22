import time
from PIL import Image
import streamlit as st
from streamlit_chat import message

from configure_llm import *
from arxiv_scraper import *
from data_pipe import *

im = Image.open("images/brand-logo-primary.jpg")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "search_title" not in st.session_state:
    st.session_state["search_title"] = None

st.set_page_config( 
                    page_title="Arxiv Search", 
                    page_icon=im, 
                    layout="wide"
                    )

st.markdown(
            "<h1 style='text-align: center;'>Arxiv-Search (Scientific Search Assistant)</h1>",
            unsafe_allow_html=True,
            )
st.divider()

if (st.session_state["search_title"] is None):
    st.markdown(
                "<h3 style='text-align: left;'>Searching Knowledge Through the Web ...</h3>",
                unsafe_allow_html=True
                )
    search_title = st.text_input("Enter the Search Title", key="SearchTitle")  
    submitted = st.button("Submit", key="SearchTitleButton")
    
    if submitted and search_title:
        st.session_state["search_title"] = search_title
        with st.spinner("Browsing the best papers..."):
            if not len(os.listdir("data/pdf")) > 0:
                scrape_papers(search_title, max_results=10)
else:
    st.markdown(
                "<h3 style='text-align: left;'>The Search Title : {}</h3>".format(st.session_state["search_title"]),
                unsafe_allow_html=True
    )
st.divider()

counter_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown(
        "<h3 style='text-align: center;'>Ask anything regarding your AI Research</h1>",
        unsafe_allow_html=True,
    )
    # st.sidebar.title("An agent that read and summarizethe the news for you")
    st.sidebar.image("images/brand-logo-primary.jpg", use_column_width=True)
    clear_conversation_button = st.sidebar.button("Clear Conversation", key="clear_conversation")
    clear_index_button = st.sidebar.button("Clear Index", key="clear_index")

    st.markdown(
    "<a style='display: block; text-align: center;' href='https://github.com/machinelearningzuu' target='_blank'> Machine Learning Zuu</a>",
    unsafe_allow_html=True,
)

if clear_conversation_button:
    st.session_state["search_title"] = None
    st.session_state["generated"] = []
    st.session_state["past"] = []

if clear_index_button:
    if len(os.listdir("data/pdf")) > 0:
        for f in os.listdir("data/pdf"):
            os.remove(f"data/pdf/{f}")

    if len(os.listdir("data/json")) > 0:
        for f in os.listdir("data/json"):
            os.remove(f"data/json/{f}")

response_container = st.container()  # container for message display

if query := st.chat_input(
    "What do you need to know? I will explain it and point you out interesting readings."
):
    st.session_state["past"].append(query)
    # try:
    with st.spinner("Reading them..."):
        index = build_index()
        query_engine = index.as_query_engine(
                                            response_mode="tree_summarize",
                                            verbose=True,
                                            similarity_top_k=5,
                                            )
    with st.spinner("Thinking..."):
        response = query_engine.query(query + credentials['llm_format_output'])

    st.session_state["generated"].append(response.response)
    del index
    del query_engine

    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True)
            message(st.session_state["generated"][i], is_user=False)

    # except Exception as e:
    #     print(e)
    #     st.session_state["generated"].append(
    #         "An error occured with the paper search, please modify your query."
    #     )