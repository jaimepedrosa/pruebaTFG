# Contenido completo del script de Python para langchain
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# --- 1. Definición de la Herramienta (Estilo Langchain/Pydantic y @tool) ---
class FinancialCalculatorInput(BaseModel):
    """Entradas para la calculadora financiera."""
    P: float = Field(description="Principal amount")
    r: float = Field(description="Annual interest rate as decimal (e.g., 0.06 for 6%)")
    n: int = Field(description="Times compounded per year")
    t: float = Field(description="Time in years")

@tool("financial_calculator", args_schema=FinancialCalculatorInput)
def financial_calculator(P: float, r: float, n: int, t: float) -> str:
    """Computes compound interest using A = P * (1 + r/n)^(n*t) and returns the result as a formatted string."""
    print(f"\n--- Ejecutando Herramienta (Langchain): financial_calculator(P={P}, r={r}, n={n}, t={t}) ---")
    result = P * (1 + r/n) ** (n*t)
    result_str = f"{result:,.2f}" # Formateado como string para el LLM
    print(f"--- Resultado Herramienta (Langchain): {result_str} ---")
    return result_str

# --- 2. Función Principal (main) ---
def main():
    model_id = "SUFE-AIFLM-Lab/Fin-R1"
    print(f"--- Iniciando experimento con langchain y modelo: {model_id} ---")

    # --- 3. Carga del Modelo y Tokenizer (Sin Cuantización) ---
    # Directriz: Carga SIN cuantización (torch.bfloat16)
    print("--- Cargando modelo y tokenizer (sin cuantización)... ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Carga en precisión original
        device_map="auto",
        trust_remote_code=True      # Requerido por Fin-R1
    )

    # Ajustar pad_token si es necesario
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("--- Modelo y tokenizer cargados ---")

    # --- 4. Creación del Pipeline (Determinista) ---
    # Directriz: Generación DETERMINISTA (do_sample=False)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"use_cache": True}, # Optimización
        # Parámetros de generación determinista
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False
            # No incluir temperature ni top_p
        }
    )
    
    # Envolver el pipeline para Langchain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("--- Pipeline de Hugging Face creado ---")

    # --- 5. Creación del Agente (LCEL) ---
    # Nota: Los modelos base necesitan un prompt claro que les indique llamar a herramientas.
    # El modelo SUFE-AIFLM-Lab/Fin-R1 no está específicamente fine-tuneado para Langchain Agent,
    # por lo que su capacidad para seguir el formato puede variar.
    tools = [financial_calculator]
    
    # Prompt que instruye al modelo sobre cómo usar las herramientas
    # Este prompt es genérico para agentes tool-calling en Langchain
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can use tools."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"), # Donde el agente pone sus pensamientos/llamadas
        ]
    )

    # Crear el agente usando la función recomendada de Langchain
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Crear el ejecutor del agente
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True muestra los pasos
    print("--- Agente Langchain creado ---")

    # --- 6. Ejecución del Query ---
    user_query = "I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years? Use the financial calculator tool."
    
    print(f"\n--- Query de Usuario: {user_query} ---")
    
    # Invocar al agente
    response = agent_executor.invoke({"input": user_query})
    
    print(f"\n--- Respuesta Final del Agente: ---")
    print(response.get("output", "No se encontró 'output' en la respuesta."))

if __name__ == "__main__":
    main()