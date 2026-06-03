"""
Streamlit Web 界面 — 知识库问答（流式版）
============================================
运行: streamlit run app.py
"""
import streamlit as st
from agent import init_knowledge, chat_stream

st.set_page_config(page_title="知识库问答 Agent", page_icon="📚")

# ============================================================
# 密码门：只有知道密码的人才能用
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 知识库问答 Agent")
    pw = st.text_input("请输入访问密码", type="password")
    if st.button("进入"):
        if pw == "2026":          # ← 改成你自己的密码
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()  # 后面全部不渲染

with st.sidebar:
    st.title("知识库问答")
    uploaded_file = st.file_uploader("📁 上传文档（TXT）", type=["txt"])
    model_choice = st.selectbox("🤖 选择模型", ["deepseek", "qwen", "local"],
        format_func=lambda x: {"deepseek":"DeepSeek V3","qwen":"通义千问","local":"本地Qwen2.5"}[x])
    search_mode = st.radio("🔍 检索模式", ["hybrid", "vector"],
        format_func=lambda x: "混合检索" if x == "hybrid" else "纯向量检索")
    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history_text = ""
        st.rerun()

st.title("知识库问答")
st.caption("上传文档 → 问任何问题 → AI 基于知识库回答")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_text" not in st.session_state:
    st.session_state.history_text = ""
if "knowledge_ready" not in st.session_state:
    st.session_state.knowledge_ready = False

if uploaded_file and not st.session_state.knowledge_ready:
    text = uploaded_file.read().decode("utf-8")
    count, topics = init_knowledge(text)
    st.session_state.knowledge_ready = True
    st.session_state.topics = topics
    st.sidebar.success(f"知识库就绪：{count} 段")

if st.session_state.knowledge_ready:
    st.sidebar.info(f"维度：{'、'.join(st.session_state.topics)}")
else:
    st.sidebar.warning("请先上传文档")
    st.info("👈 先在左侧上传 TXT 文档")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for topic, text in msg["sources"]:
                    st.caption(f"**{topic}**：{text}...")

if prompt := st.chat_input("问点什么吧..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = ""
    sources = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for chunk in chat_stream(
            prompt, st.session_state.history_text, model_choice, search_mode
        ):
            if isinstance(chunk, tuple) and chunk[0] == "__META__":
                _, reply, sources, new_history = chunk
                st.session_state.history_text = new_history
            else:
                reply += chunk
                placeholder.markdown(reply + "▌")
                import time; time.sleep(0.02)  # 30ms 延迟，让打字效果肉眼可见
        placeholder.markdown(reply)
        if sources:
            with st.expander("📎 参考来源"):
                for topic, text in sources:
                    st.caption(f"**{topic}**：{text}...")

    st.session_state.messages.append({
        "role": "assistant", "content": reply, "sources": sources,
    })
