# styles.py  ---> UI styling and CSS

import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 4px solid #1f4e79;
        }
        .user-message {
            background-color: #f0f2f6;
            border-left-color: #ff4b4b;
        }
        .assistant-message {
            background-color: #e8f4fd;
            border-left-color: #1f4e79;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1f4e79;
            margin-bottom: 1rem;
        }
        .settings-expander {
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)