# Contenido completo del script de Python para langchain
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub # Requiere 'pip install --user langchainhub'

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
    print("--- Cargando modelo y tokenizer (sin cuantización)... ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Carga en precisión original
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("--- Modelo y tokenizer cargados ---")

    # --- 4. Creación del Pipeline (Determinista) ---
    # --- CAMBIO AQUÍ: Los argumentos de generación van directos ---
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"use_cache": True}, # Optimización
        # --- Argumentos de generación determinista ---
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=model.config.pad_token_id # Aseguramos el pad_token
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("--- Pipeline de Hugging Face creado ---")

    # --- 5. Creación del Agente (LCEL - Método ReAct) ---
    tools = [financial_calculator]
    
    # Este prompt le enseña al modelo a "pensar" (Thought:) y "actuar" (Action:)
    prompt = hub.pull("hwchase17/react") 
    print("--- Prompt ReAct cargado desde Langchain Hub ---")
    
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    print("--- Agente Langchain (ReAct) creado ---")

    # --- 6. Ejecución del Query ---
    user_query = "I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years? Use the financial calculator tool."
    
    print(f"\n--- Query de Usuario: {user_query} ---")
    
    response = agent_executor.invoke({"input": user_query})
    
    print(f"\n--- Respuesta Final del Agente: ---")
    print(response.get("output", "No se encontró 'output' en la respuesta."))

if __name__ == "__main__":
    main()