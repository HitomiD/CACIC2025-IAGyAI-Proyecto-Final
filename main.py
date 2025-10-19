# Ejecutar en terminal:
# python3 clase-03/04_Rag_mozo.py


"""
Este script implementa un agente conversacional que simula ser un asistente virtual
llamado "Astor" para un equipo de desarrollo ubicado en Argentina conocido internamente como "Argentum".

Funcionalidades principales:
1.  Carga de registros del proyecto como Sprints, Incidentes, Reuniones, etc desde documentos de texto plano (incluídos en "data").
2.  Base de datos vectorial (Chroma) persistente con la información
    de dichos registros para realizar consultas semánticas.

5.  Construcción de un grafo con LangGraph para orquestar la conversación y el uso de herramientas (patrón ReAct).
6.  Un bucle interactivo para conversar con "Astor".
"""
import os
from typing import Sequence, Annotated, TypedDict, Literal

# Carga de variables de entorno
from dotenv import load_dotenv

# Componentes de LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# Componentes específicos de Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_chroma import Chroma

# Componentes de LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- 1. CONFIGURACIÓN INICIAL ---

def setup_environment():
    """Carga las variables de entorno desde el archivo .env."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    print("✅ Variables de entorno cargadas correctamente.")


# --- 2. CARGA DE DATOS DE LOS REGISTROS (Reuniones, incidentes, etc.) ---



# --- 3. CREACIÓN DEL VECTORSTORE PERSISTENTE ---

def create_or_load_vectorstore(documents: list[Document], embedding_model) -> Chroma:
    """Divide los documentos y crea o carga una base de datos vectorial Chroma persistente."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
        
    print("✅ Vectorstore listo.")
    return vectorstore


# --- 4. DEFINICIÓN DE HERRAMIENTAS ---

@tool
def off_topic_tool():
    """
    Se activa cuando el usuario pregunta algo no relacionado a los registros del proyecto o los reportes de estos registros.
    """
    return "Disculpe, solo puedo responder sobre los registros del proyecto: puede consultar sobre ellos y solicitar un análisis o la generación de un reporte."



# --- 5. DEFINICIÓN DE STATE Y AGENTES ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent_node(state: AgentState, llm):
    """Invoca al LLM con el rol de asistente para que decida el siguiente paso."""
    system_prompt = """
    Eres "Astor", el asistente del equipo de desarrollo de software "Argentum". Eres servicial, conciso y eficiente.
    Tu objetivo es ayudar al equipo a acceder a los documentos y registros del proyecto de desarrollo actual.

    Instrucciones:
    1.  Saluda al usuario y preséntate cordialmente.
    2.  Si la pregunta no tiene NINGUNA relación con los registros, el análisis de los registros o los reportes, DEBES usar la herramienta `off_topic_tool`.
    3.  Basa tus respuestas ÚNICAMENTE en la información que te proporcionan tus herramientas. No inventes registros de reuniones, incidentes ni de ningún tipo.
    4.  Sé conciso pero completo en tus respuestas.
    """
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# --- 6. LÓGICA Y CONSTRUCCIÓN DEL GRAFO (AGENTE ASISTENTE) ---


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determina si se debe llamar a una herramienta o si el flujo ha terminado."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"

def build_graph(llm_with_tools, tools_list):
    """Construye y compila el grafo del agente asistente."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    graph.add_node("tools", ToolNode(tools_list))

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    print("🧠 Grafo del asistente virtual construido y compilado.")
    return graph.compile()


# --- 7. EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    setup_environment()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key=os.getenv("GEMINI_API_KEY"), 
                                 temperature=0)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                                   google_api_key=os.getenv("GEMINI_API_KEY"))

    supervisor_tools = [off_topic_tool]
    
    supervisor_llm_with_tools = llm.bind_tools(supervisor_tools)

    multi_agent = build_graph(supervisor_llm_with_tools, supervisor_tools)

     # MODIFICACIÓN: Añadimos una lista para mantener el historial de la conversación.
    conversation_history = []
    
    print("\n\n" + "="*50)
    print("      REGISTROS DEL PROYECTO DE DESARROLLO - ARGENTUM")
    print("="*50)
    print("\n¿Te perdiste una reunión? ¿Necesitas una opinión o un reporte de lo que se rompió en el último deploy?")
    print("Astor está listo para ayudarte.")
    print(" (Escribe 'salir' para terminar la conversación)")

    while True:
        query = input("\n👤 Cliente: ")
        if query.lower() in ["exit", "quit", "salir"]:
            print("\n👋 Astor: Espero haberte ayudado, si necesitas algo mas no dudes en consultarme.")
            break
        
        # Invocamos el agente con el historial completo MÁS la nueva pregunta
        # para que el agente tenga contexto de la conversación.

        conversation_history.append(HumanMessage(content=query))
        result = multi_agent.invoke({"messages": conversation_history})
    
        # La salida del grafo (`result`) contiene el estado final, que es la lista
        # completa de mensajes de la ejecución. La guardamos como nuestro nuevo historial.
        conversation_history = result["messages"]
        
        # La respuesta para el usuario es el contenido del último mensaje en el historial.
        final_response = conversation_history[-1].content
        print(f"\n🤖 Astor: {final_response}")
