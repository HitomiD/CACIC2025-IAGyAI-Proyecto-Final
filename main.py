"""
Este script implementa un agente conversacional que simula ser un asistente virtual
llamado "Astor" para un equipo de desarrollo de nombre clave "Argentum".
Es un sistema multiagente que incluye RAG, generación de reportes con conexión a Notion
y monitoreo con langsmith.

Funcionalidades principales:
1.  Carga de registros como reuniones, sprints e incidentes.
2.  Creación de una base de datos vectorial (Chroma) persistente con la información
    de los registros para realizar consultas semánticas.
3.  Definición de un LLM (Gemini) con el rol de asistente y de generador de reportes.
4.  Integración con Notion para la persistencia de los reportes.
4.  Herramientas:
    - Un 'retriever' para buscar en los registros disponibles.
    - Una herramienta para publicar un reporte en Notion.
5.  Construcción de un grafo con LangGraph para orquestar la conversación y el uso de herramientas (patrón ReAct).
6.  Funcionamiento a partir de un bucle iterativo.
"""
import os
from typing import Sequence, Annotated, TypedDict, Literal

# Carga de variables de entorno
from dotenv import load_dotenv
from pydantic import BaseModel

# Componentes de LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool, create_retriever_tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# Componentes específicos de Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_chroma import Chroma

# Componentes de LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Dependencias para integración con Notion
from notion_client import Client 
from notion_client.errors import APIResponseError


# --- 1. CONFIGURACIÓN INICIAL ---

# Lectura de variables de entorno en .env
def setup_environment():
    """Carga las variables de entorno desde el archivo .env."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    if not os.getenv("NOTION_API_KEY") or not os.getenv("NOTION_DATABASE_ID"):
        raise ValueError("Las variables NOTION_API_KEY y/o NOTION_DATABASE_ID no están definidas.")
    print("✅ Variables de entorno cargadas correctamente.")


# --- 2. CARGA DE REGISTROS DEL PROYECTO (archivos) ---

def load_documents() -> list[Document]:
    """Carga los documentos que representan los registros del proyecto."""
    
    with open("./data/incidentes.txt", encoding='utf-8') as file:
        texto_incidentes = file.read()
    with open("./data/notas_reuniones.txt", encoding='utf-8') as file:
        texto_reuniones = file.read()
    with open("./data/sprints.txt", encoding='utf-8') as file:
        texto_sprints = file.read()

    incidentes_docs = [
        Document(
            page_content=texto_incidentes,
            metadata={"source": "incidentes.txt"}
        )
    ]
    reuniones_docs = [
        Document(
            page_content=texto_reuniones,
            metadata={"source": "notas_reuniones.txt"}
        )
    ]
    sprints_docs = [
        Document(
            page_content=texto_sprints,
            metadata={"source": "sprints.txt"}
        )
    ]

    print(f"📄 Texto extraído de los archivos.")

    return incidentes_docs + reuniones_docs + sprints_docs


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
def notion_persistence_tool(report_content: str) -> str:
    """
    Envía un reporte de texto a una base de datos de Notion.
    El contenido del reporte debe ser un resumen o análisis generado por el agente.
    """
    try: # Intento de conexión
        notion = Client(auth=os.getenv("NOTION_API_KEY"))
        database_id = os.getenv("NOTION_DATABASE_ID")
        # Ante una respuesta se crea el registro nuevo
        response = notion.pages.create(
            parent={"database_id": database_id},
            properties={
                "Name": {"title": [{"text": {"content": "Reporte de Actividad"}}]},
                "Description": {"rich_text": [{"text": {"content": report_content[:2000]}}]},  # Notion limit per block
            },
        )
        print("✅ Reporte publicado en Notion con ID:", response["id"])
        return "El reporte se ha enviado correctamente a Notion."

    except APIResponseError as e:
        print("❌ Error al enviar el reporte a Notion:", e)
        return f"Ocurrió un error al enviar el reporte a Notion: {e}"
    
# Método para definir otras herramientas y retornar la lista completa
def define_tools(vectorstore: Chroma) -> list:
    """Define las herramientas disponibles en el workflow."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Aumentamos k para más contexto
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="consultar_registros_y_notas_del_proyecto",
        description="Busca y recupera información sobre los registros de trabajo del equipo en el proyecto actual. "
    )
    print("🛠️  Herramientas del asistente definidas: consultar_registros_y_notas_del_proyecto, notion_persistence_tool.")
    return [retriever_tool, notion_persistence_tool]


# --- 5. CONSTRUCCIÓN DEL GRAFO ---

# Estado compartido del grafo
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    instruction: str

# Nodo agente RAG
def agent_node(state: AgentState, llm):
    """Invoca al LLM con el rol de agente para que recupere la información de los archivos."""
    system_prompt = """
    Eres "Astor", el asistente virtual asignado al equipo de desarrollo de software "Argentum" para el actual proyecto. Eres amable, servicial y eficiente.
    Tu objetivo es ayudar a los desarrolladores a conocer los detalles importantes del proyecto que se pudieron haber perdido.

    Instrucciones:
    1.  Saluda al desarrollador y preséntate cordialmente.
    2.  Utiliza la herramienta `consultar_registros_y_notas_del_proyecto` para responder CUALQUIER pregunta sobre minutas, registros, eventos, planificaciones o incidentes.
    3.  Si la pregunta no tiene NADA que ver con el proyecto o el equipo, DEBES responder "Solo puedo responder sobre el proyecto."`.
    4.  Basa tus respuestas ÚNICAMENTE en la información que te proporcionan tus herramientas. No inventes platos, precios ni horarios.
    5.  Sé conciso pero completo en tus respuestas, intenta no omitir demasiados detalles.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Nodo para prompts no relacionados/irrelevantes al contexto
def off_topic_node(state: AgentState) -> dict:
    """Returns a default message when the query is unrelated."""
    message = "Disculpe, como asistente solo puedo responder preguntas sobre el proyecto."
    return {"messages": state["messages"] + [HumanMessage(content=message)]}

# Nodo supervisor
def supervisor_node(state: AgentState, supervisor_llm):
    """Invoca al LLM con el rol de supervisor para que decida el siguiente paso."""
    system_prompt = """
      Eres un supervisor que determina la próxima acción a partir del prompt del usuario.
      Debes determinar si el prompt es irrelevante, requiere solo recuperar información o
      si el usuario está solicitando un reporte para enviar a Notion de manera explícita.
    Responde en formato JSON con los campos:
    - next: "rag_agent" si la consulta está relacionada con el proyecto o sus registros,
        "off_topic" si no lo está, "report_agent" si el usuario solicitó de manera EXPLÍCITA y LITERAL un reporte para enviar a NOTION.
    - instruction: un mensaje opcional que contenga cualquier instrucción para el siguiente nodo.

    """
     # Combina el prompt de sistema con el historial de mensajes
    supervisor_messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Llama al LLM supervisor con el output estructurado
    response: SupervisorOutput = supervisor_llm.invoke(supervisor_messages)

    return {
        "next": response.next,
        "instruction": response.instruction
    }

# Nodo Report Agent
def report_agent_node(state: AgentState, llm):
    """Invoca al LLM con el rol de agente de reportes para usar la herramienta de Notion."""
    system_prompt = """
    Eres un Agente de Reportes, no debes presentarte. Tu única función es generar un resumen de lo que el usuario
    solicite siempre y cuando esté relacionado a los registros del proyecto de desarrollo con
    la opción de enviarlo a Notion usando la herramienta `notion_persistence_tool`.
    Si el usuario no solicita enviar el reporte a notion de forma explícita y literal debes
    recordarle que tienes la capacidad de enviarlo si así lo pidiera.
    El contenido del reporte debe ser profesional y conciso e incluir un breve análisis al final.
    Luego de usar la herramienta, el proceso debe terminar.
    """
    # El historial completo de mensajes es el contexto para el reporte.
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Definición del enrutamiento condicional y los nodos

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determina si se debe llamar a una herramienta o si el flujo ha terminado."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"

# Enrutador para el supervisor
def router(state: AgentState) -> str:
    """Dirige el flujo al siguiente nodo basado en la decisión del supervisor."""
    if state["next"] == "rag_agent":
        return "rag_agent"
    elif state["next"] == "off_topic":
        return "off_topic_node"
    elif state["next"] == "report_agent":
        return "report_agent_node"
    else:
        return "__end__"


def build_graph(supervisor_llm, agent_llm, report_llm, tools_list):
    """Construye y compila el grafo del sistema multiagente."""
    
    # Declaración de estado a compartir
    graph = StateGraph(AgentState)

    # Añadir todos los nodos al grafo
    graph.add_node("supervisor_node", lambda state: supervisor_node(state, supervisor_llm))
    graph.add_node("agent_node", lambda state: agent_node(state, agent_llm))
    graph.add_node("report_agent_node", lambda state: report_agent_node(state, report_llm))
    graph.add_node("tools", ToolNode(tools_list))
    graph.add_node("off_topic_node", off_topic_node)


    graph.set_entry_point("supervisor_node")
    
    # Aristas condicionales del agente RAG
    graph.add_conditional_edges(
        "agent_node", 
        should_continue, 
        {"tools": "tools",
         "__end__": END}
    )

    # Aristas condicionales del agente de reporte
    graph.add_conditional_edges(
        "report_agent_node", 
        should_continue, 
        {"tools": "tools",
         "__end__": END}
    )

    # Aristas condicionales del supervisor
    graph.add_conditional_edges(
        "supervisor_node",
        router,
        {
            "rag_agent": "agent_node",
            "report_agent_node": "report_agent_node",
            "off_topic_node": "off_topic_node", 
            "__end__": END
        }
    )

    #Añadir aristas no condicionales
    graph.add_edge("tools", "agent_node") #Volver al agente luego de ejecutar herramienta
    graph.add_edge("off_topic_node", END) #Terminar ciclo al preguntar off_topic
    graph.add_edge("report_agent_node", END)

    print("🧠 Grafo del asistente virtual construido y compilado.")
    return graph.compile()



# --- 6. EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    setup_environment()
    
    # Pydantic model for supervisor output schema (required by LangChain Gemini)
    class SupervisorOutput(BaseModel):
        next: Literal["rag_agent", "report_agent", "off_topic"]
        instruction: str

    # Definicion de modelos para cada agente
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key=os.getenv("GEMINI_API_KEY"), 
                                 temperature=0)
    supervisor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key=os.getenv("GEMINI_API_KEY"), 
                                 temperature=0).with_structured_output(SupervisorOutput) #Salida estructurada en formato SupervisorOutput
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                                   google_api_key=os.getenv("GEMINI_API_KEY"))

    # Carga de banco de conocimiento
    documents = load_documents()
    vectorstore = create_or_load_vectorstore(documents, embedding_model)
    
    # Definición y binding de tools
    tools = define_tools(vectorstore)
    rag_agent_llm = llm.bind_tools(tools)
    report_agent_llm = llm.bind_tools(tools)


    rag_agent = build_graph(supervisor_llm, rag_agent_llm, report_agent_llm, tools)

    #Generar y guardar imagen en ./img
    with open("img/agent_workflow.png", "wb") as f:
        f.write(rag_agent.get_graph().draw_mermaid_png())

    
    # Lista para mantener el historial de la conversación.
    conversation_history = []
    
    print("\n\n" + "="*100)
    print("      ¿Te perdiste una reunión? ¿Necesitas saber qué se rompió en el último deploy?")
    print("="*100)
    print("\nAstor, a tu servicio por cualquier cosa que necesites saber sobre los registros del proyecto.")
    print(" (Escribe 'salir' para terminar la conversación)")

    # Bucle principal
    while True:
        query = input("\n👤 Usuario: ")
        if query.lower() in ["exit", "quit", "salir"]:
            print("\n👋 Astor: Si necesitas algo mas no dudes en consultar.")
            break
        
        # Invocamos el agente con el historial completo MÁS la nueva pregunta
        # para que el agente tenga contexto de la conversación.

        conversation_history.append(HumanMessage(content=query))
        result = rag_agent.invoke({"messages": conversation_history})
    
        # La salida del grafo (`result`) contiene el estado final, que es la lista
        # completa de mensajes de la ejecución. La guardamos como nuestro nuevo historial.
        conversation_history = result["messages"]
        
        # La respuesta para el usuario es el contenido del último mensaje en el historial.
        final_response = conversation_history[-1].content
        print(f"\n🤖 Astor: {final_response}")
