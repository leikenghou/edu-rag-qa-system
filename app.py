import os
import streamlit as st
import uuid
import shutil
import glob
import functools
from pdf_split_embeded import BGE_M3_Processor

# ==================== 初始化函数 ====================
def init_session_state():
    """初始化会话状态"""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        create_new_chat()
    
    # PDF处理相关状态
    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = BGE_M3_Processor(
            pdf_folder="./pdf_files",
            db_path="chroma_db_bge_m3",
            processed_dir="processed_pdfs"
        )
    if "pdf_processing" not in st.session_state:
        st.session_state.pdf_processing = False
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False

def create_new_chat():
    """创建新聊天"""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": f"新对话-{len(st.session_state.chats)+1}",
        "created_at": st.session_state.get("current_time", ""),
        "messages": []
    }
    st.session_state.current_chat_id = chat_id

def delete_chat(chat_id):
    """删除指定聊天"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            create_new_chat()

def clear_current_chat():
    """清空当前聊天"""
    if st.session_state.current_chat_id in st.session_state.chats:
        st.session_state.chats[st.session_state.current_chat_id]["messages"] = []

def save_uploaded_files(uploaded_files, save_dir):
    """保存上传的文件到指定目录"""
    filepaths = []
    os.makedirs(save_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        try:
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            filepaths.append(file_path)
        except Exception as e:
            st.error(f"保存文件 {uploaded_file.name} 失败: {str(e)}")
    return filepaths

# ==================== 装饰器 ====================
def check_pdf_processing_status(func):
    """检查PDF处理状态的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if st.session_state.get("pdf_processing", False):
            st.warning("PDF文件正在处理中，请稍候...")
            return
        return func(*args, **kwargs)
    return wrapper

# ==================== 聊天相关函数 ====================
@check_pdf_processing_status
def handle_send():
    """处理用户发送消息"""
    user_input = st.session_state.get("user_input_area", "").strip()
    if user_input:
        st.session_state.message_sent = True
        st.session_state.user_input_text = user_input
        st.session_state.user_input_area = ""  # 清空输入框

def generate_ai_response(model_name: str, temperature: float):
    """生成AI响应"""
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]
    
    # 从知识库获取相关内容
    try:
        collection = st.session_state.pdf_processor.client.get_collection(
            name="bge_m3_docs",
            embedding_function=st.session_state.pdf_processor.embedding_function
        )
        
        if collection.count() > 0:
            user_query = messages[-1]["content"]
            results = st.session_state.pdf_processor.query(
                query_text=user_query,
                n_results=3
            )
            
            if results["documents"]:
                context_prompt = "根据以下知识库内容回答问题:\n\n"
                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    context_prompt += f"\n相关文档 {i+1} (来自 {meta['source_file']} 第{meta['page']}页):\n{doc[:300]}...\n"
                
                messages.append({
                    "role": "system",
                    "content": context_prompt
                })
    except Exception as e:
        st.error(f"知识库查询失败: {str(e)}")
    
    # 模拟AI响应（实际应替换为您的AI模型调用）
    ai_response = "这是模拟的AI回答。实际应用中应替换为真实的AI模型生成的回答。"
    
    # 添加AI响应到聊天记录
    messages.append({
        "role": "assistant",
        "content": ai_response
    })
    
    # 更新聊天标题（如果是新对话的第一个问题）
    if len(messages) == 2:  # 用户消息 + AI响应
        current_chat["title"] = messages[0]["content"][:30] + "..."

# ==================== Streamlit UI ====================
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🎓",
    layout="wide"
)

# 初始化会话状态
init_session_state()

# 主标题
st.title("RAG智能问答系统")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("对话管理")
    
    # 显示对话列表
    chat_ids_sorted = sorted(
        st.session_state.chats.keys(),
        key=lambda x: st.session_state.chats[x]["created_at"],
        reverse=True
    )
    
    for chat_id in chat_ids_sorted:
        chat = st.session_state.chats[chat_id]
        col1, col2 = st.columns([4, 1])
        with col1:
            button_label = chat["title"]
            if chat_id == st.session_state.current_chat_id:
                button_label = f"🔍 {button_label}"
            
            if st.button(button_label, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        with col2:
            if st.button("🗑", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()
    
    if st.button("➕ 新建对话", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.divider()
    
    # ==================== 知识库管理 ====================
    st.header("知识库管理")
    
    # PDF上传和处理
    uploaded_files = st.file_uploader(
        "上传PDF文件", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # 处理参数
    with st.expander("处理参数"):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("分块大小", min_value=100, max_value=2000, value=800)
        with col2:
            chunk_overlap = st.number_input("分块重叠", min_value=0, max_value=500, value=100)
    
    # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("上传并处理", key="upload_process_btn"):
            if uploaded_files:
                # 保存文件
                save_dir = st.session_state.pdf_processor.pdf_folder
                filepaths = save_uploaded_files(uploaded_files, save_dir)
                
                if filepaths:
                    # 处理文件
                    st.session_state.pdf_processing = True
                    with st.spinner("正在处理PDF文件..."):
                        try:
                            result = st.session_state.pdf_processor.process_all_pdfs(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            
                            # 显示结果
                            st.success("处理完成!")
                            st.json({
                                "总文件数": result["total"],
                                "成功处理": result["processed"],
                                "跳过重复": result["skipped"],
                                "处理失败": result["failed"]
                            })
                            
                            # 显示详情
                            for detail in result["details"]:
                                if detail["status"] == "success":
                                    st.success(f"{detail['file']}: 成功 ({detail['chunks']}块)")
                                elif detail["status"] == "skipped":
                                    st.warning(f"{detail['file']}: 跳过 - {detail['message']}")
                                elif detail["status"] == "failed":
                                    st.error(f"{detail['file']}: 失败 - {detail['message']}")
                        
                        except Exception as e:
                            st.error(f"处理失败: {str(e)}")
                        finally:
                            st.session_state.pdf_processing = False
                else:
                    st.warning("文件保存失败")
            else:
                st.warning("请先上传PDF文件")
    
    with col2:
        if st.button("清除知识库", key="clear_knowledge_btn"):
            try:
                st.session_state.pdf_processor.client.reset()
                shutil.rmtree(st.session_state.pdf_processor.pdf_folder, ignore_errors=True)
                shutil.rmtree(st.session_state.pdf_processor.processed_dir, ignore_errors=True)
                st.success("知识库已清除")
            except Exception as e:
                st.error(f"清除失败: {str(e)}")
    
    st.divider()
    
    # ==================== 知识库状态 ====================
    st.subheader("知识库状态")
    try:
        collection = st.session_state.pdf_processor.client.get_collection(
            name="bge_m3_docs",
            embedding_function=st.session_state.pdf_processor.embedding_function
        )
        st.metric("文档数量", collection.count())
        
        if collection.count() > 0:
            with st.expander("查看文档列表"):
                unique_files = set()
                results = collection.get(include=["metadatas"])
                for meta in results["metadatas"]:
                    unique_files.add(meta["source_file"])
                
                for file in sorted(unique_files):
                    st.text(f"📄 {file}")
    except Exception as e:
        st.warning("知识库未初始化")
    
    st.divider()
    
    # ==================== 设置 ====================
    st.header("设置")
    model_name = st.selectbox(
        "选择模型",
        options=["deepseek-chat", "deepseek-reasoner"],
        index=0
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()

# ==================== 主聊天界面 ====================
chat_container = st.container()

with chat_container:
    if st.session_state.current_chat_id in st.session_state.chats:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        st.subheader(current_chat["title"])
        
        # 显示聊天历史
        for message in current_chat["messages"]:
            if isinstance(message, dict) and "role" in message and "content" in message:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    else:
        st.warning("当前聊天无效，已创建新聊天")
        create_new_chat()
        st.rerun()

# ==================== 用户输入 ====================
input_container = st.container()
with input_container:
    col1, col2 = st.columns([6, 1])
    with col1:
        st.text_area("输入您的问题...", key="user_input_area", height=100, label_visibility="collapsed")
    with col2:
        st.write("")
        st.write("")
        st.button("发送", key="send_button", on_click=handle_send, use_container_width=True)

# 处理消息发送
if st.session_state.get("message_sent", False):
    # 获取用户输入
    user_input = st.session_state.user_input_text
    
    # 重置标志
    st.session_state.message_sent = False
    
    # 显示用户消息
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 添加到聊天记录
    current_chat["messages"].append({"role": "user", "content": user_input})
    
    # 生成AI响应
    generate_ai_response(model_name=model_name, temperature=temperature)
    
    st.rerun()

# 页脚
st.markdown("---")
st.caption("© 2025 RAG智能问答系统")
