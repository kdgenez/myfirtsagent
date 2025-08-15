# utils.py

from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI


def load_llm(model_name="llama3-8b-8192", temperature=0):
    """
    Carga el modelo LLM desde OpenAI o el proveedor que uses.
    El API key se obtiene desde .streamlit/secrets.toml
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=None  # La key se carga automáticamente desde secrets
    )


def load_tools():
    """
    Carga las herramientas que usará el agente.
    Aquí puedes agregar más herramientas si lo necesitas.
    """
    search = DuckDuckGoSearchRun()
    return [search]


def build_agent(llm, tools, system_prompt):
    """
    Construye el agente ReAct con el prompt que incluye las variables necesarias.
    """
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],  # variables obligatorias
        template=f"""{system_prompt}

Tienes acceso a las siguientes herramientas:
{{tools}}

Usa el siguiente formato:

Pregunta: la pregunta del usuario
Pensamiento: tu razonamiento
Acción: la acción que tomarás
Entrada de Acción: el input de la acción
Observación: el resultado de la acción
... (repite este ciclo tantas veces como sea necesario)
Respuesta final: la respuesta final al usuario

{{input}}

{{agent_scratchpad}}
"""
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
