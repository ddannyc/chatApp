import os
import json
from typing import Optional, List

from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator
from datetime import date, datetime, timedelta
from shared import constants

from typing_extensions import Annotated
from ai_chat import generate_response, advance_generate_response

import motor.motor_asyncio


app = FastAPI(
    title="聊天机器人",
    summary="",
)
client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.my_db
message_collection = db.get_collection("chat_histories")
count_collection = db.get_collection("counts")

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


class MessageInput(BaseModel):
    user_name: str = Field(...)
    message: str = Field(...)

class MessageOutput(BaseModel):
    type: str = Field(...)
    text: str = Field(...)

class MessageModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    type: str = Field(...)
    text: str = Field(...)
    user_name: str = Field(...)
    date: datetime = Field(...)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "user_name": "Jane Doe",
                "type": "user",
                "text": "Experiments, Science, and Fashion in Nanophotonics",
                "date": datetime.now(),
            }
        },
    )

class CountModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_name: str = Field(...)
    created_at: datetime = Field(...)

class MessageCollection(BaseModel):
    messages: List[MessageOutput]

@app.post(
        "/get_ai_chat_response",
        response_description="用户输入问题，通过 ai provider 返回 ai 的回答",
)
async def get_ai_chat_response(input: MessageInput = Body(...)):
    await check_rate_limit(input.user_name)
    ai_answer = await generate_response(input.user_name, input.message)
    await count_collection.insert_one({"user_name": input.user_name, "created_at": datetime.now()})
    return {"response": ai_answer}

@app.post(
        "/get_ai_chat_response_advanced",
        response_description="在 get_ai_chat_response_advanced API 中集成情感分析模块",
)
async def get_ai_chat_response_advanced(input: MessageInput = Body(...)):
    await check_rate_limit(input.user_name)
    ai_answer = await advance_generate_response(input.user_name, input.message)
    await count_collection.insert_one({"user_name": input.user_name, "created_at": datetime.now()})
    return {"response": ai_answer}

@app.get(
        "/get_user_behavior",
        response_description="基于用户与 AI 的聊天历史，分析用户的兴趣点和行为模式"
)
async def get_user_behavior(user_name: str):
    # 聚合管道查询
    pipeline = [
        # 匹配特定用户的聊天历史
        {'$match': {'SessionId': user_name}},
    
        # 统计关键词频率
        {'$unwind': '$History'},
        {'$group': {'_id': '$History', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
        {'$limit': 10}
    ]
    
    # 执行聚合查询
    result = await message_collection.aggregate(pipeline).to_list(10)
    return result

@app.get(
        "/get_user_chat_history",
        response_description="根据输入参数输出用户的聊天记录",
        response_model=MessageCollection,
        response_model_by_alias=False,
)
async def get_user_chat_history(user_name: str, last_n: int):
    last_n_messages = await message_collection.find({"SessionId": user_name}).sort('_id', -1).limit(last_n).to_list(last_n)
    messages = [json.loads(msg["History"]) for msg in last_n_messages]
    return MessageCollection(messages=[MessageOutput(type = msg["type"], text = msg["data"]["content"]) for msg in messages])

@app.get(
        "/get_chat_status_today",
        response_description="返回用户当天聊天次数",
)
async def get_chat_status_today(user_name: str):
    count = await get_chat_count_today(user_name)
    return {"user_name": user_name, "chat_cnt": count}

async def get_chat_count_today(user_name: str):
    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    count = await count_collection.count_documents({
        "user_name": user_name,
        "created_at": {
            "$gte": start_of_day,
            "$lt": end_of_day
        }
    })
    return count

async def check_rate_limit(user_name: str):
    period_rate_limit = constants.DELTA_RATE_LIMIT_COUNT
    recent_count = await count_collection.count_documents({
        "user_name": user_name,
        "created_at": {"$gt": datetime.now() - timedelta(seconds=constants.DELTA_RATE_LIMIT)}
    })
    print(recent_count, period_rate_limit)
    if recent_count >= period_rate_limit:
        raise HTTPException(status_code=401, detail=f"每 {constants.DELTA_RATE_LIMIT} 秒最多发送 {period_rate_limit} 条信息")
    
    day_limit = constants.DAY_RATE_LIMIT_COUNT
    day_count = await get_chat_count_today(user_name)
    if day_count >= day_limit:
        raise HTTPException(status_code=401, detail=f"一天最多发送 {day_limit} 条信息")
