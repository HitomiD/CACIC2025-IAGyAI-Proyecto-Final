# Ejecutar en terminal:
# python3 clase-03/04_Rag_mozo.py


"""
Este script implementa un agente conversacional que simula ser un mozo virtual
llamado "Bruno" para el restaurante "La Delicia". Utiliza LangGraph y un sistema RAG.

Funcionalidades principales:
1.  Carga de un menú detallado y datos del restaurante como documentos.
2.  Creación de una base de datos vectorial (Chroma) persistente con la información
    del menú para realizar consultas semánticas.
3.  Definición de un LLM (Gemini 1.5 Flash) con el rol de un mozo.
4.  Herramientas:
    - Un 'retriever' para buscar en el menú.
    - Una herramienta 'off_topic' para manejar preguntas no relacionadas.
5.  Construcción de un grafo con LangGraph para orquestar la conversación y el uso de herramientas (patrón ReAct).
6.  Un bucle interactivo para chatear con "Bruno".
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

# --- 1. CONFIGURACIÓN INICIAL ---

def setup_environment():
    """Carga las variables de entorno desde el archivo .env."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    print("✅ Variables de entorno cargadas correctamente.")

# --- 2. CARGA DE DATOS DEL RESTAURANTE (MENÚ) ---

def load_documents() -> list[Document]:
    """Carga los documentos que representan el menú y la información del restaurante."""
    

    # Un solo documento con todo el menú
    menu_text = """
    Aperitivos:
    - Bruschetta Clásica: Pan tostado con tomates frescos, ajo, albahaca y aceite de oliva. Precio: $8. Ingredientes: pan, tomate, ajo, albahaca, aceite de oliva.
    - Tabla de Quesos y Fiambres: Selección de quesos locales e importados con jamón serrano y salame. Precio: $15. Ingredientes: quesos variados, jamón serrano, salame.

    Platos Principales:
    - Lomo a la Pimienta: Medallón de lomo de 250g con una cremosa salsa de pimienta negra, acompañado de puré de papas. Precio: $28. Ingredientes: lomo, pimienta, crema, puré de papas.
    - Salmón a la Parrilla con Vegetales: Filete de salmón fresco grillado con una guarnición de vegetales de estación. Precio: $25. Ingredientes: salmón, vegetales de estación.
    - Risotto de Hongos: Arroz arbóreo cremoso con una mezcla de hongos silvestres y aceite de trufa. Es un plato vegetariano. Precio: $22. Ingredientes: arroz arbóreo, hongos, aceite de trufa, queso parmesano.

    Postres:
    - Tiramisú: Clásico postre italiano con capas de bizcocho, café, mascarpone y cacao. Precio: $9. Ingredientes: bizcocho, café, queso mascarpone, cacao.
    - Volcán de Chocolate: Bizcocho tibio de chocolate con centro líquido, servido con helado de vainilla. Precio: $10. Ingredientes: chocolate, helado de vainilla.

    Bebidas:
    - Vino Malbec (copa): Vino tinto de la casa. Precio: $7.
    - Limonada con Menta y Jengibre: Bebida refrescante sin alcohol. Precio: $5.
    """
    menu_docs = [
        Document(
            page_content=menu_text,
            metadata={"source": "menu.txt"}
        )
    ]
    print(f"📄 Menú unificado en un solo documento.")


    # Un solo documento con toda la información del negocio
    negocio_info = """
    El restaurante La Delicia es propiedad de Antonio Rossi, un chef de renombre con más de 20 años de experiencia en cocina italiana.
    Ubicación: Av. Italia 1234, San Carlos de Bariloche, Río Negro, Argentina.
    La Delicia abre de martes a domingo. Horario: 12 PM – 4 PM para el almuerzo, y 8 PM – 11 PM para la cena. Lunes cerrado.
    Teléfono: +54 294 412-3456
    Email: reservas@ladelicia.com.ar
    Especialidad: Cocina italiana tradicional y platos internacionales.
    Ambiente: Familiar y acogedor, ideal para reuniones y celebraciones.
    Capacidad: 60 cubiertos.
    Se aceptan reservas y pagos con tarjeta.
    """
    info_docs = [
        Document(
            page_content=negocio_info,
            metadata={"source": "info.txt"}
        )
    ]
    print("📄 Información del negocio unificada en un solo documento.")

    return menu_docs + info_docs

# --- 3. CREACIÓN DEL VECTORSTORE PERSISTENTE ---

def create_or_load_vectorstore(documents: list[Document], embedding_model) -> Chroma:
    """Divide los documentos y crea o carga una base de datos vectorial Chroma persistente."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
        
    print("✅ Vectorstore listo.")
    return vectorstore

# --- 4. DEFINICIÓN DE HERRAMIENTAS ---

def define_tools(vectorstore: Chroma) -> list:
    """Define las herramientas que el agente mozo podrá utilizar."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Aumentamos k para más contexto
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="consultar_menu_y_horarios",
        description="Busca y recupera información sobre los platos del menú, ingredientes, precios, opciones vegetarianas, y también sobre los horarios de apertura del restaurante 'La Delicia'."
    )
    
    print("🛠️  Herramientas del mozo definidas: consultar_menu_y_horarios, off_topic_tool.")
    return [retriever_tool]

# --- 5. CONSTRUCCIÓN DEL GRAFO ---

# Estado compartido del grafo
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    instruction: str

# Nodo agente RAG
def agent_node(state: AgentState, llm):
    """Invoca al LLM con el rol de mozo para que decida el siguiente paso."""
    system_prompt = """
    Eres "Bruno", el mozo virtual del restaurante "La Delicia". Eres amable, servicial y eficiente.
    Tu objetivo es ayudar a los clientes a conocer el menú y responder sus preguntas.

    Instrucciones:
    1.  Saluda al cliente y preséntate cordialmente.
    2.  Utiliza la herramienta `consultar_menu_y_horarios` para responder CUALQUIER pregunta sobre platos, ingredientes, precios, recomendaciones y horarios.
    3.  Si el cliente te pide una recomendación (ej. "algo liviano", "un plato sin carne"), usa la herramienta para buscar opciones y luego preséntalas de forma atractiva.
    4.  Si la pregunta no tiene NADA que ver con el restaurante, el menú o la comida, DEBES usar la herramienta `off_topic_tool`.
    5.  Basa tus respuestas ÚNICAMENTE en la información que te proporcionan tus herramientas. No inventes platos, precios ni horarios.
    6.  Sé conciso pero completo en tus respuestas. Si das un precio, menciónalo claramente.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Nodo para prompts no relacionados al contexto
def off_topic_node(state: AgentState) -> dict:
    """Returns a default message when the query is unrelated."""
    message = "Disculpe, como mozo virtual de 'La Delicia', solo puedo responder preguntas sobre nuestro menú y servicios."
    return {"messages": state["messages"] + [HumanMessage(content=message)]}

# Nodo supervisor
def supervisor_node(state: AgentState, supervisor_llm):
    """Invoca al LLM con el rol de supervisor para que decida el siguiente paso."""
    system_prompt = """
      Eres un supervisor que determina si la consulta del usuario está relacionada
    con el restaurante o no.
    Responde en formato JSON con los campos:
    - next: "rag_agent" si la consulta está relacionada con el menú,
        "off_topic" si no lo está.
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


# Paso 4: Definir el enrutamiento/lógica condicional y los nodos

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
    else:
        return "__end__"


def build_graph(supervisor_llm, agent_llm, tools_list):
    """Construye y compila el grafo del sistema multiagente mozo."""
    
    # Declaración de estado a compartir
    graph = StateGraph(AgentState)

    # Añadir todos los nodos al grafo
    graph.add_node("supervisor_node", lambda state: supervisor_node(state, supervisor_llm))
    graph.add_node("agent_node", lambda state: agent_node(state, agent_llm))
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

    # Aristas condicionales del supervisor
    graph.add_conditional_edges(
        "supervisor_node",
        router,
        {
            "rag_agent": "agent_node",
            "off_topic_node": "off_topic_node", 
            "__end__": END
        }
    )

    #Añadir aristas no condicionales
    graph.add_edge("tools", "agent_node") #Volver al agente luego de ejecutar herramienta
    graph.add_edge("off_topic_node", END) #Terminar ciclo al preguntar off_topic


    print("🧠 Grafo del mozo virtual construido y compilado.")
    return graph.compile()


# --- 6. EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    setup_environment()
    
    # Pydantic model for supervisor output schema (required by LangChain Gemini)
    class SupervisorOutput(BaseModel):
        next: Literal["rag_agent", "off_topic"]
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


    rag_agent = build_graph(supervisor_llm, rag_agent_llm, tools)

    #Generar y guardar imagen en ./img
    with open("img/agent_workflow.png", "wb") as f:
        f.write(rag_agent.get_graph().draw_mermaid_png())

    
    # Lista para mantener el historial de la conversación.
    conversation_history = []
    
    print("\n\n" + "="*50)
    print("      🍝 BIENVENIDO AL RESTAURANTE 'LA DELICIA' 🍝")
    print("="*50)
    print("\nBruno, tu mozo virtual, está listo para atenderte.")
    print(" (Escribe 'salir' para terminar la conversación)")

    # Bucle principal
    while True:
        query = input("\n👤 Cliente: ")
        if query.lower() in ["exit", "quit", "salir"]:
            print("\n👋 Bruno: ¡Gracias por tu visita! ¡Vuelve pronto!")
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
        print(f"\n🤖 Bruno: {final_response}")
