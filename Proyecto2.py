import streamlit as st
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
import os

# Cargar las variables de entorno
load_dotenv()


def main():
    st.set_page_config(page_title="Agente CSV y Python", layout="wide")
    st.title("Agente CSV y Python")

    # Instrucciones para el usuario
    st.markdown("""
    ### Instrucciones:
    - Seleccione una solicitud de trabajo para el Python Agent y presione el botón **Ejecutar Python**.
    - Ingrese una pregunta para el agente y presione el botón **Procesar Pregunta**.
    - El agente decidirá automáticamente si utiliza el Python Agent o el CSV Agent según corresponda.
    """)

    # Agregar estilos personalizados
    st.markdown("""
        <style>
        .stApp {background-color: #f0f0f0;}
        .stButton > button {background-color: #4CAF50; color: white; border-radius: 5px;}
        </style>
    """, unsafe_allow_html=True)

    # Opciones de solicitudes para el Python Agent
    python_tasks = [
        "Genera un programa que calcule el factorial de un número",
        "Diseña un programa que convierta una cantidad en dólares a euros, pesos mexicanos y yenes.",
        "Diseña un programa que convierta temperaturas de Celsius a Fahrenheit y viceversa."
    ]

    # Menú de selección
    selected_task = st.selectbox("Seleccione una solicitud para el Python Agent:", python_tasks)

    # Botón para ejecutar Python Agent
    if st.button("Ejecutar Python"):
        st.write("Ejecutando Python Agent...")
        python_response = python_agent_executor.invoke({"input": selected_task})
        st.markdown("### Respuesta del Python Agent:")
        st.code(python_response["output"], language="python")

    # Campo de texto para preguntas
    user_query = st.text_area("Ingrese su pregunta para el agente (sobre CSV o Python):")

    # Botón para procesar preguntas
    if st.button("Procesar Pregunta"):
        st.write("Procesando su pregunta con el Agent...")
        if user_query.strip():
            response = grand_agent_executor.invoke({"input": user_query})
            st.markdown("### Respuesta del Agent:")
            st.write(response["output"])
        else:
            st.error("Por favor, ingrese una pregunta válida.")


# Configuración de agentes
instructions = """
You are an agent designed to write and execute Python code or analyze CSV files to answer questions.
You can decide which tool to use based on the query:
- Use the Python Agent for programming tasks or Python-related queries.
- Use the CSV Agent for questions about the provided CSV files.
"""

# Crear agentes individuales
tools = []

# Crear el agente de Python
base_prompt = hub.pull("langchain-ai/react-agent-template")
python_prompt = base_prompt.partial(instructions=instructions)
python_agent = create_react_agent(
    prompt=python_prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=[PythonREPLTool()],
)
python_agent_executor = AgentExecutor(agent=python_agent, tools=[PythonREPLTool()], verbose=True)

# Crear el agente de CSV
csv_files = ["netflix_titles.csv", "books.csv", "music.csv", "vgsales.csv"]  # Reemplace con sus archivos reales
for csv_file in csv_files:
    if not os.path.exists(csv_file):
        st.warning(f"Archivo {csv_file} no encontrado. Asegúrese de colocarlo en el directorio de trabajo.")
csv_agent = create_csv_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    path=csv_files[0],  # Puede cambiarse dinámicamente si se requiere
    verbose=True,
    allow_dangerous_code=True,
)

# Crear herramientas y el Grand Agent
tools.append(
    Tool(
        name="Python Agent",
        func=python_agent_executor.invoke,
        description="Use this tool for Python-related tasks."
    )
)
tools.append(
    Tool(
        name="CSV Agent",
        func=csv_agent.invoke,
        description="Use this tool for analyzing CSV files."
    )
)

grand_agent = create_react_agent(
    prompt=python_prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=tools,
)
grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

if __name__ == "__main__":
    main()