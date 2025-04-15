import os
import streamlit as st
import functions 
import uuid


# ==================== streamlit开头配置 ====================
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🎓",
    layout="wide"  # 使用宽布局以适应侧边栏
)



# ==================== 页面设计 ====================
# 初始化会话状态
functions.init_session_state()

st.title("RAG智能问答系统")


# 侧边栏配置
with st.sidebar:
    st.header("对话管理")
    # 显示现有对话列表
    chat_ids_sorted = sorted(
        st.session_state.chats.keys(),
        key=lambda x: st.session_state.chats[x]["created_at"],
        reverse=True
    )
    
    for chat_id in chat_ids_sorted:
        chat = st.session_state.chats[chat_id]
        col1, col2 = st.columns([4, 1])
        with col1:
            # 使用唯一键以避免冲突
            button_key = f"chat_button_{chat_id}"
            # 为当前活动聊天添加标记
            button_label = chat["title"]
            if chat_id == st.session_state.current_chat_id:
                button_label = f"🔍 {button_label}"
            
            if st.button(button_label, key=button_key, use_container_width=True):
                st.session_state.current_chat_id = chat_id
                # 强制重新加载页面
                st.experimental_rerun()
        
        with col2:
            if st.button("🗑", key=f"delete_{chat_id}"):
                functions.delete_chat(chat_id)
                # 强制重新加载页面
                st.experimental_rerun()
    
    if st.button("➕ 新建对话", use_container_width=True):
        functions.create_new_chat()
        st.experimental_rerun()
    
    st.divider()
    
    st.header("设置")
    model_name = st.selectbox(
        "选择模型",
        options=["deepseek-chat", "deepseek-reasoner"],
        index=0
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()

    # 上传pdf文件按钮
    st.sidebar.subheader("知识库 PDF 上传")
    uploaded_pdf_files = st.sidebar.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=True)
    #  本地存储 PDF 文件的目录
    UPLOAD_FOLDER = "pdf文件"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True) # 确保目录存在

    if st.sidebar.button("上传pdf文件"): 
        if uploaded_pdf_files: # 只有当有上传文件时才执行保存
            filepaths = functions.save_uploaded_files(uploaded_pdf_files, UPLOAD_FOLDER) # 调用保存函数
            if filepaths:
                st.success(f"成功保存 {len(filepaths)} 个 PDF 文件到 '{UPLOAD_FOLDER}' 目录！")
            else:
                st.error("保存 PDF 文件失败，请检查。")
        else:
            st.warning("请先上传 PDF 文件。")

        

    
     # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 清空当前对话", on_click=functions.clear_current_chat, use_container_width=True):
            st.experimental_rerun()
    with col2:
        if st.session_state.get("is_streaming", True):
            if st.button("⏸️ 终止生成", use_container_width=True):
                st.session_state.is_streaming = False
        else:
            if st.button("▶️ 恢复生成", use_container_width=True):
                st.session_state.is_streaming = True


    # 聊天导入导出功能
    if st.button("💾 导出聊天记录", use_container_width=True):
        if functions.save_chats_to_file():
            st.success("聊天记录已保存")

    uploaded_file = st.file_uploader("导入聊天记录", type=["yaml"], key="chat_uploader")
    if uploaded_file is not None:
        # 添加确认按钮，避免自动处理
        if st.button("确认加载此文件"):
            # 创建唯一的临时文件名
            temp_file_path = f"temp_upload_{uuid.uuid4()}.yaml"
            try:
                # 保存上传的文件
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 加载聊天记录
                load_success = functions.load_chats_from_file(temp_file_path)
                
                # 显示结果消息
                if load_success:
                    st.success("聊天记录已导入")
                    # 使用单次重新加载，避免循环
                    st.experimental_rerun()
                
            except Exception as e:
                st.error(f"处理上传文件时出错: {e}")
            finally:
                # 确保在任何情况下都删除临时文件
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        st.warning(f"无法删除临时文件 {temp_file_path}: {e}")
    
    st.divider()


   
# 创建容器以控制UI元素顺序
chat_container = st.container()

# 在聊天容器中显示当前对话
with chat_container:
    if st.session_state.current_chat_id in st.session_state.chats:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        st.subheader(current_chat["title"])
        
        # 确保消息列表存在且有效
        if "messages" in current_chat and isinstance(current_chat["messages"], list):
            # 显示聊天历史
            for message in current_chat["messages"]:
                if isinstance(message, dict) and "role" in message and "content" in message:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        else:
            st.warning("当前聊天没有有效的消息记录")
    else:
        # 如果当前ID无效，创建新聊天
        st.warning("当前聊天ID无效，已创建新聊天")
        functions.create_new_chat()
        st.experimental_rerun()

# 在输入容器中放置输入框和发送按钮
input_container = st.container()  # 输入框容器
with input_container:
    col1, col2 = st.columns([6, 1])
    with col1:
        # 使用text_area替代chat_input
        st.text_area("输入您的问题...", key="user_input_area", height=100)

    with col2:
        # 添加一些垂直间距，使按钮垂直居中
        st.write("")
        st.write("")
        # 添加发送按钮，使用on_click回调
        st.button("发送", key="send_button", on_click=functions.handle_send, use_container_width=True)

# 处理消息发送逻辑
if st.session_state.get("message_sent", False):
    # 获取用户输入
    user_input = st.session_state.user_input_text
    # 重置标志
    st.session_state.message_sent = False
    
    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 检查最后一条消息的角色
    last_message_role = None
    if current_chat["messages"]:
        last_message_role = current_chat["messages"][-1]["role"]
    
    # 如果最后一条消息也是用户消息，则合并它们
    if last_message_role == "user":
        current_chat["messages"][-1]["content"] += "\n\n" + user_input
    else:
        # 添加用户消息到历史记录
        current_chat["messages"].append({"role": "user", "content": user_input})
    
    # 生成AI响应
    functions.generate_ai_response(model_name=model_name, temperature=temperature)

# 页脚
st.markdown("---")
st.caption("© 2025 RAG智能问答系统")
