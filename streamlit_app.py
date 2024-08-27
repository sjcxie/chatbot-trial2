import streamlit as st
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Show title and description.
st.title("💬 Cocobot evaluation study")
st.write(
    "This is a cocobot prototype solely for evaluation purposes. "
    "To use this app, you need to provide an OpenAI API key, which is provided to you. "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# # Load environment variables from the .env file
# load_dotenv('my.env')

# # Get the OpenAI API key from the environment variable
# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OpenAI_Key"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

else:

     # Create an OpenAI client.
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    
    # system and human prompts
    # read system prompt
    # Specify the path to your .txt file
    file_path = 'system_prompt.txt'

    # Open the file and read its contents into a string variable
    with open(file_path, 'r') as file:
        system_prompt_text = file.read()
    system_message_template = system_prompt_text
    human_message_template = """{human_input}"""
    system_prompt_template = SystemMessagePromptTemplate.from_template(template=system_message_template)
    human_prompt_template = HumanMessagePromptTemplate.from_template(template=human_message_template)
    chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_prompt_template,
        MessagesPlaceholder(variable_name="chat_history"), # dynamic insertion of past conversation history
        human_prompt_template,
    ]
    )
    # st.info(system_prompt_text)

    # set up memory buffer for the unadabot
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, llm=llm)
    
    memory.clear()
    # create a chatbot llm chain
    botchain = LLMChain(
        llm=llm,
        prompt=chat_prompt_template,
        verbose=False,
        output_parser=StrOutputParser(),
        memory=memory,
    )

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hello there! How can I assist you today?"},
        ]

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if user_input := st.chat_input("Enter your input here."):
        
        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate a response using the OpenAI API.
        botchain.predict(human_input=user_input)
        bot_response = memory.buffer[-1].content

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        df = pd.DataFrame(st.session_state.messages)
        csv = df.to_csv(index=False)
        
        # Let the user know how many turns they have completed
        st.info("You have completed {} of turn so far.".format(int(df.shape[0]/2)))

        if (df.shape[0]/2 > 5):
            st.download_button(
            "Download chat history", 
            data = csv,
            file_name="messages.csv",
            mime="text/csv",
            )

        if (df.shape[0]/2 > 10):
            st.info("Don't forget to click 'Download chat history' to save a copy of the chat history and send it to the research team.")
        