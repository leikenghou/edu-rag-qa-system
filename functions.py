import streamlit as st
import uuid
import datetime
import yaml
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from pdf_split_embeded import BGE_M3_Processor
from typing import List, Optional, Dict

# ==================== 环境变量 ====================
load_dotenv()
api_key = os.getenv('API_KEY')
base_url = os.getenv('BASE_URL')

if api_key is None or base_url is None:
    missing_vars = []
    if api_key is None:
        missing_vars.append('API_KEY')
    if base_url is None:
        missing_vars.append('BASE_URL')
    raise ValueError(f"环境变量缺失: {', '.join(missing_vars)}")

# 初始化客户端
client = OpenAI(api_key=api_key, base_url=base_url)

TOKENIZERS_PARALLELISM=False

# ==================== 会话状态初始化 ====================
def init_session_state():
    """初始化会话状态"""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    
    if "current_chat_id" not in st.session_state:
        create_new_chat()
    
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = True
    
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False
    
    # 初始化PDF处理器
    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = BGE_M3_Processor(
            pdf_folder="./pdf_files",
            db_path="chroma_db_bge_m3",
            processed_dir="processed_pdfs"
        )
    
    # 知识库状态
    if "knowledge_base_ready" not in st.session_state:
        st.session_state.knowledge_base_ready = False

# ==================== 聊天管理函数 ====================
def create_new_chat():
    """创建新的对话"""
    new_chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chats[new_chat_id] = {
        "title": f"对话 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "messages": [],
        "created_at": datetime.datetime.now()
    }

def delete_chat(chat_id: str):
    """删除指定对话"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if chat_id == st.session_state.current_chat_id:
            if st.session_state.chats:
                latest_chat_id = sorted(
                    st.session_state.chats.keys(),
                    key=lambda x: st.session_state.chats[x]["created_at"],
                    reverse=True
                )[0]
                st.session_state.current_chat_id = latest_chat_id
            else:
                create_new_chat()

def clear_current_chat():
    """清空当前聊天"""
    if st.session_state.current_chat_id in st.session_state.chats:
        st.session_state.chats[st.session_state.current_chat_id]["messages"] = []

# ==================== 文件操作函数 ====================
def save_chats_to_file() -> bool:
    """保存聊天记录到YAML文件"""
    try:
        chat_data = {
            chat_id: {
                "title": chat["title"],
                "messages": chat["messages"],
                "created_at": chat["created_at"].isoformat()
            }
            for chat_id, chat in st.session_state.chats.items()
        }
        
        os.makedirs("chats", exist_ok=True)
        filename = f"chats/chats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(chat_data, f, allow_unicode=True)
        return True
    except Exception as e:
        st.error(f"保存聊天记录失败: {e}")
        return False

def load_chats_from_file(file_path: str) -> bool:
    """从YAML文件加载聊天记录"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chat_data = yaml.safe_load(f)
        
        st.session_state.chats = {}
        sorted_chats = []
        
        for chat_id, chat in chat_data.items():
            if isinstance(chat, dict) and all(k in chat for k in ["created_at", "messages", "title"]):
                try:
                    chat["created_at"] = datetime.datetime.fromisoformat(chat["created_at"])
                    # 验证消息格式
                    valid_messages = [
                        msg for msg in chat["messages"] 
                        if isinstance(msg, dict) and "role" in msg and "content" in msg
                        and msg["role"] in ["user", "assistant", "system"]
                    ]
                    chat["messages"] = valid_messages
                    sorted_chats.append((chat_id, chat))
                except (ValueError, TypeError):
                    continue
        
        sorted_chats.sort(key=lambda x: x[1]["created_at"], reverse=True)
        
        for chat_id, chat in sorted_chats:
            st.session_state.chats[chat_id] = chat
        
        if st.session_state.chats:
            st.session_state.current_chat_id = sorted_chats[0][0]
            return True
        else:
            create_new_chat()
            return False
    except Exception as e:
        st.error(f"加载聊天记录失败: {e}")
        create_new_chat()
        return False

def save_uploaded_files(uploaded_files: List, upload_folder: str) -> Optional[List[str]]:
    """保存上传的PDF文件到指定目录"""
    if not uploaded_files:
        return None
    
    saved_filepaths = []
    os.makedirs(upload_folder, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        try:
            filepath = os.path.join(upload_folder, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_filepaths.append(filepath)
        except Exception as e:
            st.error(f"保存文件 '{uploaded_file.name}' 失败: {e}")
            continue
    
    return saved_filepaths if saved_filepaths else None

# ==================== PDF处理函数 ====================
def process_uploaded_pdfs(chunk_size: int = 800, chunk_overlap: int = 100) -> Dict:
    """处理所有上传的PDF文件"""
    try:
        result = st.session_state.pdf_processor.process_all_pdfs(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        st.session_state.knowledge_base_ready = result["processed"] > 0
        return result
    except Exception as e:
        st.error(f"处理PDF文件失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def query_knowledge_base(query_text: str, n_results: int = 3) -> Optional[Dict]:
    """查询PDF知识库"""
    if not st.session_state.knowledge_base_ready:
        return None
    
    try:
        return st.session_state.pdf_processor.query(
            query_text=query_text,
            n_results=n_results
        )
    except Exception as e:
        st.error(f"查询知识库失败: {e}")
        return None

# ==================== AI响应生成 ====================
def generate_ai_response(model_name: str = "deepseek-chat", temperature: float = 0.7):
    """生成AI回复，整合PDF知识库内容"""
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]
    
    # 确保最后一条是用户消息
    if not messages or messages[-1]["role"] != "user":
        st.warning("需要用户消息才能生成回复")
        return
    
    # 从知识库获取相关内容
    knowledge_context = ""
    if st.session_state.knowledge_base_ready:
        knowledge_results = query_knowledge_base(messages[-1]["content"])
        if knowledge_results and knowledge_results["documents"]:
            knowledge_context = "\n\n相关背景知识:\n"
            for i, (doc, meta) in enumerate(zip(knowledge_results["documents"][0], 
                                              knowledge_results["metadatas"][0])):
                knowledge_context += f"\n[{i+1}] 来自文档 '{meta['source_file']}' (第{meta['page']}页):\n{doc[:300]}...\n"
    
    # 构建系统提示
    system_prompt = {
        "role": "system",
        "content": f"""你是一个有用的AI助手。请基于以下信息回答问题:
        {knowledge_context if knowledge_context else '无相关背景知识'}
        如果不知道答案，请直接说"我不知道"。不要编造信息。"""
    }
    
    # 准备消息历史
    processed_messages = [system_prompt]
    last_role = None
    
    for msg in messages:
        if last_role == msg["role"]:
            processed_messages[-1]["content"] += "\n\n" + msg["content"]
        else:
            processed_messages.append({"role": msg["role"], "content": msg["content"]})
            last_role = msg["role"]
    
    # 流式生成响应
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=processed_messages,
                temperature=temperature,
                stream=True
            )
            
            for chunk in response:
                if not st.session_state.get("is_streaming", True):
                    st.warning("生成已停止")
                    break
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            current_chat["messages"].append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"生成回复时出错: {e}")

# ==================== 消息处理 ====================
def handle_send():
    """处理用户发送的消息"""
    user_input = st.session_state.get("user_input_area", "").strip()
    if user_input:
        st.session_state.user_input_text = user_input
        st.session_state.user_input_area = ""
        st.session_state.message_sent = True
        
        # 添加到当前聊天记录
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        current_chat["messages"].append({"role": "user", "content": user_input})
        
        # 如果是新对话的第一个问题，更新标题
        if len(current_chat["messages"]) == 1:
            current_chat["title"] = user_input[:30] + ("..." if len(user_input) > 30 else "")
