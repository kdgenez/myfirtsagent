**app.py**
```python
# app.py
from __future__ import annotations
from typing import List
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from utils import init_llm, build_agent, available_tools

st.set_page_config(page_title="Llama3 Agent • LangChain ✨", page_icon="🤖", layout="wide")
st.title("🤖 Llama 3 Agent (LangChain + Streamlit)")

secrets = st.secrets
app_cfg = secrets.get("app", {})
MODEL = app_cfg.get("model", "llama3-8b-8192")
DEFAULT_SYSTEM_PROMPT = app_cfg.get("default_system_prompt", "You are a helpful, concise assistant.")
DEFAULT_TEMPERATURE = float(app_cfg.get("temperature", 0.2))
DEFAULT_TOP_P = float(app_cfg.get("top_p", 0.95))

GROQ_API_KEY = secrets.get("GROQ_API_KEY")
TAVILY_API_KEY = secrets.get("TAVILY_API_KEY")
if not GROQ_API_KEY:
    st.error("No se encontró GROQ_API_KEY en st.secrets.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuración del modelo")
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT)
    temperature = st.slider("Temperature", 0.0, 1.0, float(DEFAULT_TEMPERATURE), 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, float(DEFAULT_TOP_P), 0.05)
    st.divider()
    st.subheader("🧰 Herramientas")
    use_tavily = st.toggle("Tavily Web Search", value=bool(TAVILY_API_KEY))
    use_calculator = st.toggle("Calculadora", value=True)
    st.caption("Modelo: "+MODEL)

llm = init_llm(api_key=GROQ_API_KEY, model_name=MODEL, temperature=temperature, top_p=top_p)
tools = available_tools(use_tavily=use_tavily and bool(TAVILY_API_KEY), use_calculator=use_calculator, tavily_api_key=TAVILY_API_KEY)
agent_executor = build_agent(llm=llm, tools=tools, system_prompt=system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages: List[BaseMessage] = []

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Escribe tu mensaje…")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                result = agent_executor.invoke({"input": user_input, "chat_history": st.session_state.messages[:-1]})
                output_text = result.get("output", str(result))
            except Exception as e:
                output_text = f"⚠️ Error: {e}"
            st.markdown(output_text)
            st.session_state.messages.append(AIMessage(content=output_text))
```

---
