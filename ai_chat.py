import os
from shared import constants
from langchain.schema import (
    HumanMessage,
)
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompt_values import ChatPromptValue
from langchain.memory import MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# LLM
load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(
        temperature=0.7,
        # model=constants.OPENROUTER_DEFAULT_CHAT_MODEL,
        # openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        # openai_api_base=constants.OPENROUTER_API_BASE,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
    )
def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    messages = prompt.to_messages()
    num_tokens = llm.get_num_tokens_from_messages(messages)
    ai_function_messages = messages[2:]
    while num_tokens > 8_096:
        ai_function_messages = ai_function_messages[2:]
        num_tokens = llm.get_num_tokens_from_messages(
            messages[:2] + ai_function_messages
        )
    messages = messages[:2] + ai_function_messages
    return ChatPromptValue(messages=messages)
chain = prompt | condense_prompt | llm

async def generate_response(session_id: str, input_text: str):
    memory = get_memory(session_id)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory,
        input_messages_key="question",
        history_messages_key="history",
    )
    print(memory.messages)
    config = {"configurable": {"session_id": session_id}}
    resp = await chain_with_history.ainvoke({"question": input_text}, config=config)
    return resp.content

def get_memory(session_id: str):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://localhost",
        database_name="my_db",
        collection_name="chat_histories",
    )

# print(generate_response("test_session", "Hi! I'm Alan"))
# print(generate_response("Alan", "Hi! I am Alan"))
# print(generate_response("Alan", "Whats my name"))
# print(generate_response("Bob", "Hi! I am Bob"))
# print(generate_response("Bob", "Whats my name"))
