from datetime import datetime
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompt_values import ChatPromptValue
from langchain.memory import MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# LLM
load_dotenv()


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
    while num_tokens > 2_096:
        ai_function_messages = ai_function_messages[2:]
        num_tokens = llm.get_num_tokens_from_messages(
            messages[:2] + ai_function_messages
        )
    messages = messages[:2] + ai_function_messages
    return ChatPromptValue(messages=messages)

async def generate_response(session_id: str, input_text: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | condense_prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_memory(session_id),
        input_messages_key="question",
        history_messages_key="history",
    )
    config = {"configurable": {"session_id": session_id}}
    resp = await chain_with_history.ainvoke({"question": input_text}, config=config)
    return resp.content

async def advance_generate_response(session_id: str, input_text: str):
    emotion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "根据<消息>的内容判断情感倾向是<正面>、<负面>还是<中性>, 你回答的内容只包含判断结果"),
            ("human", "消息: {question}; 当前情感倾向是? 注意你的回答仅包含情感倾向选项。"),
        ]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位得力的助手，你将根据用户的<情感倾向>调整回复的语气和内容，以更好地适应用户的情绪状态。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "<情感倾向>: {emotion}; {question}，回复的内容中**不要包含**<情感倾向>相关的信息"),
        ]
    )
    chain = prompt | condense_prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_memory(session_id),
        input_messages_key="question",
        history_messages_key="history",
    )
    emotion = await (emotion_prompt | llm).ainvoke({"question": input_text})
    print(emotion)
    config = {"configurable": {"session_id": session_id}}
    resp = await chain_with_history.ainvoke({"question": input_text, "emotion": emotion.content}, config=config)
    return resp.content

def get_memory(session_id: str):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://localhost",
        database_name="my_db",
        collection_name="chat_histories",
    )
