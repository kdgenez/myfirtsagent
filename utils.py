**utils.py**
```python
from __future__ import annotations
from typing import List, Optional
import ast, operator as op
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None

def init_llm(api_key: str, model_name: str = "llama3-8b-8192", temperature: float = 0.2, top_p: float = 0.95) -> BaseChatModel:
    return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature, top_p=top_p)

_ALLOWED_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg}

def _eval_expr(node):
    if isinstance(node, ast.Num): return node.n
    if isinstance(node, ast.BinOp): return _ALLOWED_OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp): return _ALLOWED_OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("Operación no permitida")

def calculator_run(expression: str) -> str:
    try:
        return str(_eval_expr(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error en calculadora: {e}"

calculator_tool = Tool(name="calculator", func=calculator_run, description="Calculadora aritmética básica.")

def available_tools(use_tavily: bool, use_calculator: bool, tavily_api_key: Optional[str] = None) -> List[Tool]:
    tools: List[Tool] = []
    if use_calculator:
        tools.append(calculator_tool)
    if use_tavily and TavilySearchResults and tavily_api_key:
        tools.append(TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key))
    return tools

def build_agent(llm: BaseChatModel, tools: List[Tool], system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history", optional=True), ("human", "{input}"), ("system", "Piensa paso a paso.")])
    return AgentExecutor(agent=create_react_agent(llm=llm, tools=tools, prompt=prompt), tools=tools, verbose=False, handle_parsing_errors=True)
```

---
