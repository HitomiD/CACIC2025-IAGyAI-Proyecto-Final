import os
from typing import TypedDict, Annotated, List
from pydantic import BaseModel
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv # Usa dotenv para cargar las variables de entorno

load_dotenv()

# --- Configuración Inicial (Opcional: Carga tus claves de API) ---
# os.environ["OPENAI_API_KEY"] = "TU_CLAVE_DE_OPENAI"
# os.environ["GOOGLE_API_KEY"] = "TU_CLAVE_DE_GOOGLE"

# Paso 1: Definir el Estado del Grafo
# El estado es el objeto central que se pasa entre todos los nodos.
# Mapea directamente a las variables de estado ($flow.state) en Flowise.

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    next: str
    instruction: str

# Pydantic model for supervisor output schema (required by LangChain Gemini)
class SupervisorOutput(BaseModel):
    next: str
    instruction: str

# Paso 2: Crear los Nodos de Agente y el Nodo Final
# Creamos una función genérica para los agentes trabajadores para no repetir código.
def create_agent_node(llm, system_prompt: str):
    """Crea un nodo de agente que ejecuta una tarea con un prompt de sistema específico."""
    def agent_node(state: AgentState) -> dict:
        # El agente recibe la instrucción del supervisor
        task_message = HumanMessage(content=state["instruction"])

        # Ejecuta el LLM con el prompt de sistema y la tarea actual
        response = llm.invoke([SystemMessage(content=system_prompt), task_message])

        # Devuelve el resultado para que se añada al historial de mensajes
        return {"messages": [response]}
    return agent_node

# Nodo para el Ingeniero de Software
software_agent_node = create_agent_node(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.9),
    """As a Senior Software Engineer, you are a pivotal part of our innovative development team. Your expertise and leadership drive the creation of robust, scalable software solutions.
    Your goal is to lead the development of high-quality software solutions.
    Design and implement new feature for the given task, ensuring it integrates seamlessly with existing systems. Use your understanding of React, Tailwindcss, NodeJS to build this feature.
    The output should be a fully functional, well-documented feature. Include detailed comments in the code."""
)

# Nodo para el Revisor de Código
code_reviewer_agent_node = create_agent_node(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.9),
    """As a Quality Assurance Engineer, you are an integral part of our development team, ensuring our software products are of the highest quality.
    Your goal is to ensure the delivery of high-quality software through thorough code review and testing.
    Review the codebase for the new feature designed and implemented by the Senior Software Engineer. Provide constructive feedback, guiding contributors towards best practices."""
)

# Nodo para la respuesta final
def final_answer_node(state: AgentState) -> dict:
    """Genera la respuesta final consolidando toda la conversación."""
    final_prompt = """Given the above conversations, generate a detail solution developed by the software engineer and code reviewer.
    Your guiding principles:
    1. Preserve Full Context: Include all code implementations, improvements and review from the conversation. Do not omit, summarize, or oversimplify key information.
    2. Markdown Output Only: Your final output must be in Markdown format."""

    # Añade el prompt final al historial de mensajes
    final_messages = state["messages"] + [HumanMessage(content=final_prompt)]

    # Llama al modelo de Google para la respuesta final
    final_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.9)
    response = final_llm.invoke(final_messages)

    return {"messages": [response]}

# Paso 3: Crear el Nodo Supervisor
# Este nodo utiliza "structured output" para forzar al LLM a devolver un JSON,
# replicando la funcionalidad del supervisor en Flowise.
supervisor_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.9
).with_structured_output(SupervisorOutput)

def supervisor_node(state: AgentState) -> dict:
    """Decide el siguiente paso y da instrucciones."""
    system_prompt = """You are a supervisor tasked with managing a conversation between the following workers:
    - Software Engineer
    - Code Reviewer
    Given the user request and the conversation history, respond with the worker to act next and provide specific instructions for them.
    When the task is complete, respond with FINISH."""

    # Combina el prompt de sistema con el historial de mensajes
    supervisor_messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Llama al LLM supervisor
    response = supervisor_llm.invoke(supervisor_messages)

    # Devuelve la decisión para actualizar el estado
    return {
        "next": response.next,
        "instruction": response.instruction
    }

# Paso 4: Definir el Enrutamiento Condicional
# Esta función actúa como el nodo "Condition" de Flowise.
def router(state: AgentState) -> str:
    """Dirige el flujo al siguiente nodo basado en la decisión del supervisor."""
    if state["next"] == "SOFTWARE":
        return "software_agent"
    elif state["next"] == "REVIEWER":
        return "code_reviewer_agent"
    else:
        return "final_answer"

# Paso 5: Construir y Compilar el Grafo
# Aquí unimos todos los nodos y bordes para crear el flujo de trabajo.
workflow = StateGraph(AgentState)

# Añadir los nodos al grafo
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("software_agent", software_agent_node)
workflow.add_node("code_reviewer_agent", code_reviewer_agent_node)
workflow.add_node("final_answer", final_answer_node)

# Establecer el punto de entrada
workflow.set_entry_point("supervisor")

# Añadir los bordes condicionales
workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "software_agent": "software_agent",
        "code_reviewer_agent": "code_reviewer_agent",
        "final_answer": "final_answer"
    }
)

# Añadir los bordes que crean el ciclo (loop)
# Después de que un agente termina, vuelve al supervisor.
workflow.add_edge("software_agent", "supervisor")
workflow.add_edge("code_reviewer_agent", "supervisor")

# El nodo final termina el grafo
workflow.add_edge("final_answer", END)

# Compilar el grafo en una aplicación ejecutable
app = workflow.compile()

with open("clase-04/imagen/multiagent.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

# --- Cómo Ejecutar el Código ---
initial_task = "Create a simple React component that displays a button. When clicked, it should show an alert with the message 'Hello, World!'."

# La entrada inicial para el grafo es el primer mensaje del usuario.
inputs = {"messages": [HumanMessage(content=initial_task)]}

# Ejecutar el grafo y mostrar los resultados de cada paso
for s in app.stream(inputs, {"recursion_limit": 10}):
    if "__end__" not in s:
        print(s)
        print("----")
