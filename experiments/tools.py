import os
from google.colab import userdata
import gradio as gr
import requests
import inspect
import pandas as pd
from typing import TypedDict, Annotated
from huggingface_hub import InferenceClient, login, list_models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
#from langchain.schema import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.docstore.document import Document
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
import datasets
import re
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
import wikipediaapi, wikipedia
from duckduckgo_search import DDGS
import yt_dlp
import cv2
import glob
import subprocess

HUGGINGFACEHUB_API_TOKEN = userdata.get('HF_TOKEN')

login(token=HUGGINGFACEHUB_API_TOKEN, add_to_git_credential=True)

llm = HuggingFaceEndpoint(
    #repo_id="HuggingFaceH4/zephyr-7b-beta",
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    timeout=240,
)


class State(TypedDict):
    question: str
    article_content: str
    answer: str
    messages: Annotated[list, add_messages]

def should_fetch_wikipedia(question):
    """
    Determines whether to fetch Wikipedia content based on the question's content.

    Args:
        question: The question string.

    Returns:
        True if Wikipedia content should be fetched, otherwise False.
    """

    # List of keywords that might indicate a Wikipedia-related question
    wikipedia_keywords = [
        "explain",
        "define",
        "describe",
        "history",
        "background",
        "facts",
        "information",
        "summary",
        "overview",
        "wiki",
        "wikipedia"  # Explicitly checking for "wikipedia" in the question
    ]

    # Compile a regular expression for case-insensitive keyword matching
    keyword_pattern = re.compile(r"|".join(wikipedia_keywords), re.IGNORECASE)

    # Check if any of the keywords are present in the question
    if keyword_pattern.search(question):
        return True  # Fetch Wikipedia content
    else:
        return False  # Answer directly using the LLM
    

def answer_directly(state: State) -> State:
    # Directly invoke the LLM with the question
    response = llm.invoke([{"role": "user", "content": state['question']}])
    return {
        **state,
        "answer": response,
        "messages": state["messages"] + [{"role": "assistant", "content": response}]
    }


def filter_wikipedia_urls(url_list):
    # Pattern matches any Wikipedia domain (including language subdomains)
    wikipedia_pattern = re.compile(r'https?://([a-z]{2,3}\.)?wikipedia\.org/')
    return [url for url in url_list if wikipedia_pattern.search(url)]

def get_wikipedia_content_from_url(url):
    # Extract the article title from the URL
    # e.g., https://en.wikipedia.org/wiki/Python_(programming_language)
    title = url.split("/wiki/")[-1]
    # Replace underscores with spaces for the wikipedia library
    title = title.replace("_", " ")
    # Fetch the page content
    page = wikipedia.page(title)
    return page.content

def get_wikipedia_article_full_text(search_query):
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(search_query)
        if not search_results:
            return None
        page = wikipedia.page(search_results[0])
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.page(e.options[0]).content
    except wikipedia.exceptions.PageError:
        return None

def fetch_wikipedia_content(state: State) -> State:
    try:
        article_content = get_wikipedia_article_full_text(state["question"])
        if not article_content:
            raise ValueError("No Wikipedia article found")
            
        return {
            **state,
            "article_content": article_content,
            "messages": state["messages"] + [{"role": "system", "content": "Wikipedia article fetched."}]
        }
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [{"role": "system", "content": f"Error: {str(e)}"}]
        }


def answer_from_article(state: State) -> State:
    # Compose a prompt for the LLM
    prompt = f"Based on the following Wikipedia article, answer the question:\n\nArticle:\n{state['article_content']}\n\nQuestion: {state['question']}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {
        **state,
        "answer": response,
        "messages": state["messages"] + [{"role": "assistant", "content": response}]
    }