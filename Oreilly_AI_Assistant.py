# -*- coding: utf-8 -*-

!pip install langchain openai pandas streamlit python-dotenv langchain-experimental langchain-openai langchain-core langchain_community --q

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv, set_key
import openai
from openai import OpenAI
#from openai.error import AuthenticationError
import os
import pandas as pd
import streamlit as st


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_1" not in st.session_state:
        st.session_state.df_1 = None
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False  # Track API key validation


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def get_openai_api_key():
    # Ask user for OpenAI API Key in the Streamlit interface
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if st.button("Set API Key") and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        set_key(".env", "OPENAI_API_KEY", api_key)
        st.session_state.api_key = api_key
        st.success("API Key set successfully! Verifying...")
        return api_key
    return None


def verify_openai_api_key(api_key):
    try:
        openai.api_key = api_key  # Set the key temporarily
        client = OpenAI(api_key = openai.api_key)
        # Perform a dummy ChatCompletion request
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "ping"}]
        )
        st.success("API Key is valid!")
        st.session_state.api_key_valid = True
        return True
    # except openai.error.AuthenticationError:
    #     st.error("Invalid API Key. Please try again.")
    #     st.session_state.api_key_valid = False
    #     return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.session_state.api_key_valid = False
        return False


def create_agent(df, df_1):
    # Ensure pandas doesn't truncate the display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )

    # Create the agent with modified configuration
    return create_pandas_dataframe_agent(
        ChatOpenAI(
            temperature=0, model_name="gpt-4"
        ),
        [df, df_1],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=10
    )


def main():
    load_dotenv()  # Load environment variables
    initialize_session_state()

    # Check for OpenAI API Key
    if not st.session_state.api_key_valid:
        api_key = get_openai_api_key()
        if api_key and verify_openai_api_key(api_key):
            st.query_params = {"validated": "true"}
            st.session_state.api_key_valid = True
            st.rerun()  # Reload the app after successful key validation
        else:
            st.stop()  # Prevent further execution until the key is valid


    # Customize initial app landing page
    st.set_page_config(page_title="Shopwise AI Assistant", page_icon="🤖")
    st.title("Welcome! I am your Shopwise AI Assistant 🤖")

    # File uploaders in sidebar
    with st.sidebar:
        st.header("Upload Data")
        product_file = st.file_uploader("Upload Product CSV", type="csv")
        order_file = st.file_uploader("Upload Order CSV", type="csv")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Process uploaded files
    if product_file and order_file:
        if st.session_state.df is None or st.session_state.df_1 is None:
            # Load complete datasets
            st.session_state.df = pd.read_csv(product_file)
            st.session_state.df_1 = pd.read_csv(order_file)

            # Display dataset info
            st.sidebar.write(f"Product DataFrame: {len(st.session_state.df)} rows")
            st.sidebar.write(f"Order DataFrame: {len(st.session_state.df_1)} rows")

            agent = create_agent(st.session_state.df, st.session_state.df_1)
            st.session_state.agent = agent
            st.success(f"Data loaded successfully!")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("How can I assist you?"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Get agent response
                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):
                        stream_handler = StreamHandler(st.empty())
                        # Include chat history context in the prompt
                        context = "\n".join([f"{m['role']}: {m['content']}"
                                             for m in st.session_state.messages[:-1]])
                        full_prompt = f"""Chat history:\n{context}\n\nCurrent question: {prompt}
                        IMPORTANT: Analyze the COMPLETE dataset using df.iloc[:] or df.loc[:]. Do not use head() or tail()."""

                        response = st.session_state.agent.run(full_prompt, callbacks=[stream_handler])
                        st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if "API Connection" in str(e):
                    st.warning("Please check your OpenAI API key.")
    else:
        st.info("Please upload both Product and Order CSV files to begin.")


if __name__ == "__main__":
    main()
