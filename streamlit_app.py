import streamlit as st
from openai import OpenAI
import pandas as pd
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials

# Show title and description.
st.title("💬 Cocobot Evaluation Study")
st.write(
    "This is a cocobot prototype solely for evaluation purposes. "
    "First enter the participant ID that is provided to you."
    "After you complete a chat with the cocobot, please don't forget to enter either 'save' or 'stop' ."
)

if user_PID := st.text_input("What is your participant ID?"):
    st.info("Thank you for providing your participant ID.")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

    openai_api_key = st.secrets["API_KEY"]
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
            {"role": "assistant", "content": "Hello there! How are you feeling today?"},
        ]

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat_history_df = pd.DataFrame(st.session_state.messages)

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if user_input := st.chat_input("Enter your input here. Enter 'SAVE' or 'STOP' if you want to stop the conversation."):
        
        if user_input=="SAVE" or user_input=="save" or user_input=="STOP" or user_input=="stop":
            st.write("The button was clicked. Running code to upload chat history.")
        
            file_name = "Chat_History_P{PID}.csv".format(PID=user_PID)
            st.write("file name is "+file_name)
            chat_history_df.to_csv(file_name, index=False)
            credentials_dict = {
            'type': st.secrets.gcs["type"],
            'client_id': st.secrets.gcs["client_id"],
            'client_email': st.secrets.gcs["client_email"],
            'private_key': st.secrets.gcs["private_key"],
            'private_key_id': st.secrets.gcs["private_key_id"],
            }
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                credentials_dict
            )
            client = storage.Client(credentials=credentials, project='galvanic-fort-430920-e8')
            bucket = client.get_bucket('streamlit-bucket-bot-eval')
            blob = bucket.blob(file_name)
            blob.upload_from_filename(file_name)
            st.write("Chat history was uploaded. You can safely exit this chat now.")
        
        else:

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

            chat_history_df = pd.DataFrame(st.session_state.messages)
            
            # Let the user know how many turns they have completed
            st.info("You have completed {} of turn so far.".format(int(chat_history_df.shape[0]/2)))
            
            if (chat_history_df.shape[0]/2 > 8):
                st.info("""When you are ready to stop the conversation, you can enter "SAVE" to upload the chat history and conclude this session.""")

        
        
        
            

            
