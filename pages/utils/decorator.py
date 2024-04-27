import streamlit as st
import traceback
from functools import wraps

def with_progress_bar(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with st.spinner("运行中..."):
            
            result = func(*args, **kwargs)
        return result
    return wrapper
