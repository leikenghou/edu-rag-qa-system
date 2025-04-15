import streamlit as st
import uuid
import datetime
import yaml
import os
from openai import OpenAI
from dotenv import load_dotenv


# ==================== 环境变量 ====================
# 加载环境变量
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

# ==================== 工具函数 ====================
def init_session_state():
    """初始化会话状态"""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    
    if "current_chat_id" not in st.session_state:
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chats[new_chat_id] = {
            "title": f"对话 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "messages": [],
            "created_at": datetime.datetime.now()
        }
    
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = True
    
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False



def create_new_chat():
    """创建新的对话"""
    new_chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chats[new_chat_id] = {
        "title": f"对话 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "messages": [],
        "created_at": datetime.datetime.now()
    }


def switch_chat(chat_id):
    """切换到指定对话"""
    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):
    """删除指定对话"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        # 如果删除的是当前对话，则切换到最新的对话或创建新对话
        if chat_id == st.session_state.current_chat_id:
            if st.session_state.chats:
                # 按创建时间排序并取最新的对话
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
        # 可以选择添加一个系统消息作为初始上下文
        # st.session_state.chats[st.session_state.current_chat_id]["messages"].append({
        #     "role": "system", 
        #     "content": "你是一个有用的AI助手。请简洁明了地回答用户问题。"
        # })


def save_chats_to_file():
    """保存聊天记录到文件"""
    chat_data = {
        chat_id: {
            "title": chat["title"],
            "messages": chat["messages"],
            "created_at": chat["created_at"].isoformat()
        }
        for chat_id, chat in st.session_state.chats.items()
    }
    
    os.makedirs("chats", exist_ok=True)
    with open(f"chats/chats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(chat_data, f, allow_unicode=True)
    
    return True


def load_chats_from_file(file_path):
    """从文件加载聊天记录"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chat_data = yaml.safe_load(f)
        
        # 清空当前所有聊天记录
        st.session_state.chats = {}
        
        # 按创建时间排序
        sorted_chats = []
        for chat_id, chat in chat_data.items():
            if isinstance(chat, dict) and "created_at" in chat and "messages" in chat and "title" in chat:
                chat["created_at"] = datetime.datetime.fromisoformat(chat["created_at"])
                sorted_chats.append((chat_id, chat))
        
        # 按创建时间降序排序
        sorted_chats.sort(key=lambda x: x[1]["created_at"], reverse=True)
        
        # 验证并加载每个聊天记录
        for chat_id, chat in sorted_chats:
            # 验证消息格式
            valid_messages = []
            for msg in chat["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if msg["role"] in ["user", "assistant", "system"]:
                        valid_messages.append(msg)
            
            # 更新聊天记录
            chat["messages"] = valid_messages
            st.session_state.chats[chat_id] = chat
        
        # 如果有加载的聊天记录，则设置当前聊天为最新的聊天
        if st.session_state.chats:
            latest_chat_id = sorted_chats[0][0] if sorted_chats else None
            if latest_chat_id:
                st.session_state.current_chat_id = latest_chat_id
                st.success(f"成功加载 {len(st.session_state.chats)} 个聊天记录")
                return True  # 删除了这里的 st.experimental_rerun()
            else:
                st.warning("未找到有效的聊天记录")
                create_new_chat()  # 没有有效聊天记录时创建新聊天
        else:
            st.warning("未找到有效的聊天记录")
            create_new_chat()  # 没有有效聊天记录时创建新聊天
        
        return True
    except Exception as e:
        st.error(f"加载聊天记录失败: {e}")
        # 确保创建一个新聊天，避免应用崩溃
        if "chats" not in st.session_state or not st.session_state.chats:
            create_new_chat()
        return False


def generate_ai_response(model_name="deepseek-chat", temperature=0.7):
    """生成AI回复"""
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # 处理消息以确保用户和助手消息交替
            processed_messages = []
            last_role = None
            
            for msg in current_chat["messages"]:
                # 如果当前消息与上一条消息的角色相同，合并它们
                if last_role == msg["role"]:
                    processed_messages[-1]["content"] += "\n\n" + msg["content"]
                else:
                    processed_messages.append({"role": msg["role"], "content": msg["content"]})
                    last_role = msg["role"]
            
            # 确保最后一条消息是用户消息
            if processed_messages and processed_messages[-1]["role"] != "user":
                st.warning("需要用户输入才能生成回复")
                return
            
            # 流式调用
            response = client.chat.completions.create(
                model=model_name,
                messages=processed_messages,
                temperature=temperature,
                stream=True
            )
            
            for chunk in response:
                if not st.session_state.get("is_streaming", True):
                    st.warning("流式传输已停止")
                    break
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"生成回复时出错: {e}")
            return
        
        # 将助手回复添加到当前聊天记录
        current_chat["messages"].append({"role": "assistant", "content": full_response})


# 处理发送消息的回调函数
def handle_send():
    if st.session_state.user_input_area.strip():
        # 将输入内容保存到临时变量
        st.session_state.user_input_text = st.session_state.user_input_area
        # 清除输入框
        st.session_state.user_input_area = ""
        # 设置标志表示需要处理消息
        st.session_state.message_sent = True



def save_uploaded_files(uploaded_files, upload_folder):
    """保存上传的 PDF 文件到本地文件夹"""
    saved_filepaths = []
    if not uploaded_files:
        return None # 或者返回空列表 []，表示没有文件上传
    for uploaded_file in uploaded_files:
        filepath = os.path.join(upload_folder, uploaded_file.name)
        try:
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_filepaths.append(filepath)
        except Exception as e:
            print(f"保存文件 '{uploaded_file.name}' 失败: {e}")
            return None # 或者可以选择继续处理其他文件，并返回部分成功的文件路径列表
    return saved_filepaths # 返回成功保存的文件路径列表