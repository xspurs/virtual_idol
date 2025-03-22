#!/usr/bin/python3

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import uuid
from datetime import datetime
import os

# -------------------- 配置部分 --------------------
DEEPSEEK_API_KEY = os.environ.get("API_KEY")
MODEL_NAME = "deepseek-reasoner"

CELEBRITY_PROFILES = {
    "univu5": {
        "data_file": "data_univu5.json",
        "index_file": "index_univu5.bin",
        "user_avatar": "👤",  # 可用本地路径如"./avatars/user.png"
        "bot_avatar": "🎤",   # 或在线URL
    },
    "周杰伦": {
        "data_file": "data_jay.json",
        "index_file": "index_jay.bin",
        "user_avatar": "👤",  # 可用本地路径如"./avatars/user.png"
        "bot_avatar": "🎤",   # 或在线URL
    }
}

# 基础设置
st.set_page_config(
    page_title="虚拟偶像",
    # page_icon="./images/favicon.ico"
)

# -------------------- 会话状态初始化 --------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_star" not in st.session_state:
    st.session_state.selected_star = None
if "processing" not in st.session_state:  # 新增：跟踪处理状态
    st.session_state.processing = False
if "chats" not in st.session_state:
    st.session_state.chats = {}  # 格式：{chat_id: {history: [], star: "周杰伦"}}

# -------------------- 模型初始化 --------------------
@st.cache_resource
def load_retriever():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

retriever = load_retriever()

# -------------------- 数据加载函数 --------------------
def load_star_data(star_name):
    """加载指定明星的数据和索引"""
    config = CELEBRITY_PROFILES[star_name]
    with open(config["data_file"], "r") as f:
        json_data = json.load(f)
    # index = faiss.read_index(config["index_file"])
    vectors = retriever.encode([item["text"] for item in json_data])
    index = faiss.IndexFlatL2(vectors.shape[1])
    return json_data, index

# -------------------- 生成回答函数（分离处理逻辑）-----------------
def generate_response(user_input: str):
    """独立处理生成逻辑的函数"""
    # 1. 检索上下文
    json_data, index = load_star_data(st.session_state.selected_star)
    query_vector = retriever.encode([user_input])
    distances, indices = index.search(query_vector, 3)
    context = "\n".join([json_data[i]["text"] for i in indices[0]])

    # 2. 调用API生成回答
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"你现在的身份是{st.session_state.selected_star}，请根据以下信息回答：{context}，回答要口语化，符合人设，避免输出“（歪着头思考）”等带括号形式的文字，避免和文字聊天不相符，避免每句话都带口头禅。"},
                {"role": "user", "content": user_input}
            ]
        }
    ).json()
    
    # 3. 添加机器人消息到历史
    st.session_state.history.append({
        "role": "assistant",
        "content": response["choices"][0]["message"]["content"]
    })
    
    # 4. 清除处理状态
    st.session_state.processing = False

def create_new_chat(star_name):
    """创建新聊天会话"""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "history": [],
        "star": star_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.active_chat_id = chat_id

# -------------------- 前置选择页 --------------------
if not st.session_state.selected_star:
    st.title("请选择对话明星")
    selected = st.selectbox(
        "选择要对话的明星",
        list(CELEBRITY_PROFILES.keys()),
        index=0
    )
    
    if st.button("开始对话"):
        st.session_state.selected_star = selected
        st.rerun()

# -------------------- 主聊天页 --------------------
else:
    current_star = CELEBRITY_PROFILES[st.session_state.selected_star]
    json_data, index = load_star_data(st.session_state.selected_star)
    
    
    st.title(f"{st.session_state.selected_star} >")
    
    # 侧边栏设置
    # with st.sidebar:
    #     st.header("聊天会话")
        
    #     # 新建聊天按钮
    #     if st.button("+ 新建聊天"):
    #         create_new_chat(list(CELEBRITY_PROFILES.keys())[0])
    #         st.rerun()
        
    #     # 聊天会话列表
    #     st.subheader("历史会话")
    #     for chat_id in list(st.session_state.chats.keys())[-5:]:  # 显示最近5个
    #         chat = st.session_state.chats[chat_id]
    #         emoji = CELEBRITY_PROFILES[chat["star"]]["bot_avatar"]
    #         if st.button(
    #             f"{emoji} {chat['star']} - {chat['created_at']}",
    #             key=chat_id,
    #             use_container_width=True
    #         ):
    #             st.session_state.active_chat_id = chat_id
    #             st.rerun()
    
    # 实时消息渲染（先渲染已有历史记录）
    for message in st.session_state.history:
        avatar = current_star["bot_avatar"] if message["role"] == "assistant" else current_star["user_avatar"]
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # 处理用户输入（分步执行）
    user_input = st.chat_input("输入你的问题...", disabled=st.session_state.processing)
    
    if user_input and not st.session_state.processing:
        # 步骤1：立即添加并显示用户消息
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.processing = True  # 锁定输入
        
        # 步骤2：立即重新渲染界面（此时会先显示用户消息）
        st.rerun()  # 触发脚本重新执行
        
    elif st.session_state.processing:
        # 步骤3：在重新渲染后执行生成操作
        last_user_input = st.session_state.history[-1]["content"]
        
        # 使用占位符显示加载状态
        with st.status(f"{st.session_state.selected_star}正在输入中...", expanded=True) as status:
            # st.write(f"正在咨询{st.session_state.selected_star}...")
            generate_response(last_user_input)
            status.update(label="生成完成", state="complete", expanded=False)
        
        # 步骤4：清除处理状态并刷新
        st.session_state.processing = False
        st.rerun()