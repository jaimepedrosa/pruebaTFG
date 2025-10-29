# Contenido completo del script de Python para langchain (Versión Simplificada LCEL)
import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

# --- Función para parsear la llamada a la herramienta ---
def parse_tool_call(llm_output: str) -> dict | None:
    """Extrae el JSON de la llamada a la herramienta de la salida del LLM."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        tool_call_json = match.group(1).strip()
        try:
            return json.loads(tool_call_json)
        except json.JSONDecodeError:
            print("Error: No se pudo decodificar el JSON de la llamada a la herramienta.")
            return None
    else:
        print("Advertencia: No se encontró <tool_call> en la salida del LLM.")
        return None

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
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"use_cache": True},
        # Argumentos de generación determinista
        max_new_tokens=150, # Suficiente para la llamada a la herramienta
        do_sample=False,
        pad_token_id=model.config.pad_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("--- Pipeline de Hugging Face creado ---")

    # --- 5. Creación de la Cadena LCEL (Flujo Simplificado) ---
    
    # Prompt para generar la LLAMADA a la herramienta
    tool_call_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that must use the financial_calculator tool.
Available tool:
- financial_calculator(P: float, r: float, n: int, t: float): compute compound interest using A = P * (1 + r/n)^(n*t)

Output ONLY the tool call in the following JSON format wrapped in <tool_call> tags:
<tool_call>
{{"tool": "financial_calculator", "input": {{"P": <principal>, "r": <annual_rate>, "n": <times_compounded_per_year>, "t": <years>}}}}
</tool_call>"""),
        ("user", "{input}")
    ])

    # Prompt para generar la RESPUESTA FINAL usando el resultado de la herramienta
    final_response_prompt_template = ChatPromptTemplate.from_messages([
         ("system", "You are a helpful assistant. You have received the result from a tool."),
         ("user", "{original_input}\n\nTool Result for financial_calculator: {tool_result}"),
         ("ai", "Based on the calculation, the final result is:") # Ayuda al modelo a empezar
    ])

    # --- Cadena LCEL Completa ---
    # 1. Generar la llamada a la herramienta
    # 2. Parsear la salida para obtener el JSON
    # 3. Ejecutar la herramienta con el JSON
    # 4. Formatear el prompt final con la pregunta original y el resultado
    # 5. Generar la respuesta final
    
    chain = (
        # Paso 1 y 2: Generar y Parsear Llamada
        RunnablePassthrough.assign(
            tool_call_json_str = tool_call_prompt_template | llm | StrOutputParser()
        )
        | RunnablePassthrough.assign(
            tool_call_dict = lambda x: parse_tool_call(x["tool_call_json_str"])
        )
        # Paso 3: Ejecutar Herramienta (si la llamada fue parseada)
        | RunnablePassthrough.assign(
             tool_result = lambda x: financial_calculator.invoke(x["tool_call_dict"]["input"]) if x["tool_call_dict"] else "Error: Tool call failed."
        )
        # Paso 4: Preparar entradas para el prompt final
        | RunnablePassthrough.assign(
            final_prompt_input = lambda x: {
                "original_input": x["input"], 
                "tool_result": x["tool_result"]
            }
        )
        # Paso 5: Generar Respuesta Final
        | RunnablePassthrough.assign(
             final_answer = final_response_prompt_template | llm | StrOutputParser()
        )
    )
    print("--- Cadena LCEL creada ---")

    # --- 6. Ejecución del Query ---
    user_query = "I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years?"
    
    print(f"\n--- Query de Usuario: {user_query} ---")
    
    # Invocar la cadena completa
    response_dict = chain.invoke({"input": user_query})
    
    print(f"\n--- Salida del LLM (Turno 1 - Tool Call): ---")
    print(response_dict.get("tool_call_json_str", "No se generó llamada."))
    print(f"\n--- Resultado de la Herramienta: ---")
    print(response_dict.get("tool_result", "La herramienta no se ejecutó."))
    print(f"\n--- Respuesta Final del Agente (Turno 2): ---")
    print(response_dict.get("final_answer", "No se generó respuesta final."))

if __name__ == "__main__":
    main()