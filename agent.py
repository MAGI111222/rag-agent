"""
Agent 核心：知识库 + 检索 + LLM + 工具（纯函数，不依赖 UI）
"""
import os, jieba, logging, requests, urllib3
from datetime import datetime
from pymilvus import MilvusClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载密钥，不暴露在代码里
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

jieba.setLogLevel(logging.WARNING)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MILVUS_DB = "./milvus.db"
COLLECTION = "knowledge"
DIM = 768
CHUNK_SIZE = 500      # 每块 500 字
CHUNK_OVERLAP = 60    # 块之间重叠 60 字


# ============================================================
# Embedding
# ============================================================
def embed(text: str) -> list[float]:
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
    )
    return resp.json()["embedding"]


# ============================================================
# 知识库：通用切分 → 向量化 → Milvus
# ============================================================
def init_knowledge(text: str):
    """传入任意全文 → 自动切分 → 入库。"""
    global MILVUS_DB, COLLECTION

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", "."],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        chunks = [text]

    db = MilvusClient(MILVUS_DB)

    # 如果表已存在就用现成的，不存在才建（彻底避免文件锁冲突）
    if not db.has_collection(COLLECTION):
        db.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            metric_type="COSINE",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
    else:
        # 表已存在，清空旧数据
        try:
            all_ids = [r["id"] for r in db.query(collection_name=COLLECTION, filter="", output_fields=["id"])]
            if all_ids:
                db.delete(collection_name=COLLECTION, ids=all_ids)
        except Exception:
            pass

    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "id": i,
            "vector": embed(chunk),
            "text": chunk,
            "topic": f"第{i+1}段",
        })

    db.insert(collection_name=COLLECTION, data=data)
    # 确保表已加载就绪，否则搜索会报 "released" 错
    try:
        db.load_collection(COLLECTION)
    except Exception:
        pass
    return len(data), [f"{len(data)}块文档"]


# ============================================================
# 混合检索
# ============================================================
def keyword_score(query: str, text: str) -> float:
    q_words = set(jieba.cut(query))
    t_words = set(jieba.cut(text))
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words | t_words)


def search(query: str, top_k: int = 3, mode: str = "hybrid") -> list[dict]:
    """
    检索。mode="hybrid" → 向量 + 关键词混合
         mode="vector" → 纯向量
    返回 [{"text": ..., "topic": ..., "score": ...}, ...]
    """
    alpha = 1.0 if mode == "vector" else 0.6
    db = MilvusClient(MILVUS_DB)
    q_vec = embed(query)

    ann = db.search(
        collection_name=COLLECTION,
        data=[q_vec],
        limit=top_k * 3,
        output_fields=["text", "topic"],
    )

    candidates = []
    for item in ann[0]:
        txt = item["entity"]["text"]
        topic = item["entity"].get("topic", "未知")
        vec_sim = 1.0 - item["distance"]
        kw_sim = keyword_score(query, txt)
        candidates.append({
            "text": txt,
            "topic": topic,
            "score": alpha * vec_sim + (1 - alpha) * kw_sim,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# 模型工厂：根据选择返回不同的 LLM 客户端
# ============================================================
def get_model(choice: str = "deepseek"):
    if choice == "qwen":
        return ChatOpenAI(
            api_key=os.getenv("QWEN_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.7,
        )
    elif choice == "local":
        return ChatOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen2.5:1.5b",
            temperature=0.7,
        )
    else:  # deepseek
        return ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_KEY", ""),
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.7,
        )


# ============================================================
# 工具
# ============================================================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持 + - * / ** 运算。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"

@tool
def get_weather(city: str) -> str:
    """查询城市实时天气。"""
    KEY = os.getenv("AMAP_KEY", "")
    geo = requests.get(
        "https://restapi.amap.com/v3/config/district",
        params={"key": KEY, "keywords": city, "subdistrict": 0},
        verify=False,
    ).json()
    if geo["status"] != "1":
        return f"找不到城市：{city}"
    adcode = geo["districts"][0]["adcode"]
    w = requests.get(
        "https://restapi.amap.com/v3/weather/weatherInfo",
        params={"key": KEY, "city": adcode, "extensions": "base"},
        verify=False,
    ).json()
    if w["status"] != "1":
        return f"查不到天气"
    live = w["lives"][0]
    return f"{live['city']}：{live['weather']}，{live['temperature']}°C，湿度{live['humidity']}%"

@tool
def get_current_time(dummy: str = "") -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

TOOL_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_current_time": get_current_time,
}


# ============================================================
# 一轮对话（供 Streamlit 调用）
# ============================================================
def chat(
    user_input: str,
    history_text: str = "",
    model_choice: str = "deepseek",
    search_mode: str = "hybrid",
):
    """
    一轮对话。返回 (reply: str, sources: list, updated_history: str)
    """
    model = get_model(model_choice)
    model_with_tools = model.bind_tools([calculator, get_weather, get_current_time])

    # RAG 检索
    results = search(user_input, top_k=3, mode=search_mode)
    context = "\n".join(
        f"- [{r['topic']}] {r['text'][:200]}".encode("gbk", errors="ignore").decode("gbk")
        for r in results
    )
    sources = [(r["topic"], r["text"][:150]) for r in results]

    # System Prompt：把对话记忆放在最前面，LLM 优先看到
    sys_content = f"""你是 Amadeus，一个智能助手。

【规则】
1. 优先用知识库资料回答，提及出自哪个维度
2. 资料里没有就说知识库中暂无相关信息，禁止编造
3. 可用工具：calculator、get_weather、get_current_time
4. 用自己的话总结，不要照抄原文

【知识库资料】
{context if context else "（未搜到相关资料）"}

{history_text if history_text else ""}"""

    sys_content = sys_content.encode("gbk", errors="ignore").decode("gbk")
    ui_clean = user_input.encode("gbk", errors="ignore").decode("gbk")

    messages = [SystemMessage(content=sys_content), HumanMessage(content=ui_clean)]
    msg = model_with_tools.invoke(messages)

    # 工具调用
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        tc = msg.tool_calls[0]
        func_name = tc['name']
        func_args = tc['args']

        tool_func = TOOL_MAP[func_name]
        result = tool_func.invoke(func_args) if hasattr(tool_func, 'invoke') else tool_func(**func_args)

        messages.append(msg)
        messages.append(ToolMessage(content=result, tool_call_id=tc['id']))
        final = model_with_tools.invoke(messages)
        reply = final.content
    else:
        reply = msg.content

    reply = reply.encode("gbk", errors="ignore").decode("gbk")

    # 更新记忆，按"用户:"计数保留最近 8 轮
    new_history = history_text + f"用户: {ui_clean}\nAmadeus: {reply}\n"
    # 统计有多少轮对话（数 "用户:" 出现的次数）
    rounds = new_history.count("用户:")
    if rounds > 8:
        # 找到第 (rounds-8) 个 "用户:" 的位置，从那里截断
        idx = -1
        for _ in range(rounds - 8):
            idx = new_history.index("用户:", idx + 1)
        new_history = "【之前对话已省略】\n" + new_history[idx:]

    return reply, sources, new_history


# ============================================================
# 流式版本：边生成边返回（SSE）
# ============================================================
def chat_stream(user_input, history_text="", model_choice="deepseek", search_mode="hybrid"):
    """
    跟 chat() 逻辑一样，但用 yield 逐字返回。
    调用方用 for chunk in chat_stream(...): 接收。
    """
    model = get_model(model_choice)
    model_with_tools = model.bind_tools([calculator, get_weather, get_current_time])

    # RAG 检索
    yield "🔍 正在检索..."
    results = search(user_input, top_k=3, mode=search_mode)
    yield f"找到 {len(results)} 条资料\n"
    context = "\n".join(
        f"- [{r['topic']}] {r['text'][:200]}".encode("gbk", errors="ignore").decode("gbk")
        for r in results
    )
    sources = [(r["topic"], r["text"][:150]) for r in results]

    sys_content = f"""你是 Amadeus，一个智能助手。

【规则】
1. 优先用知识库资料回答
2. 资料里没有就说知识库中暂无相关信息，禁止编造
3. 可用工具：calculator、get_weather、get_current_time
4. 用自己的话总结

【知识库资料】
{context if context else "（未搜到相关资料）"}

{history_text if history_text else ""}"""

    sys_content = sys_content.encode("gbk", errors="ignore").decode("gbk")
    ui_clean = user_input.encode("gbk", errors="ignore").decode("gbk")
    messages = [SystemMessage(content=sys_content), HumanMessage(content=ui_clean)]

    # 第一轮调用（检查工具）
    msg = model_with_tools.invoke(messages)
    full_reply = ""

    # 处理工具调用
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        tc = msg.tool_calls[0]
        func_name = tc['name']
        func_args = tc['args']
        yield f"[调用工具: {func_name}]\n"

        tool_func = TOOL_MAP[func_name]
        result = tool_func.invoke(func_args) if hasattr(tool_func, 'invoke') else tool_func(**func_args)
        messages.append(msg)
        messages.append(ToolMessage(content=result, tool_call_id=tc['id']))

        # 流式输出最终回答
        for chunk in model_with_tools.stream(messages):
            if chunk.content:
                full_reply += chunk.content
                yield chunk.content
    else:
        full_reply = msg.content
        yield msg.content

    # 更新记忆并返回元数据（最后一条特殊 yield 带标记）
    full_reply = full_reply.encode("gbk", errors="ignore").decode("gbk")
    new_history = history_text + f"用户: {ui_clean}\nAmadeus: {full_reply}\n"
    yield ("__META__", full_reply, sources, new_history)
