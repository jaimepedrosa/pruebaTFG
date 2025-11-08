# Contenido completo del script de Python para langchain (Versión Corregida)
import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# NOTA: Se eliminan las importaciones de StoppingCriteria, ya que usaremos 'stop_sequences'

# --- 1. Definición de la Herramienta ---
class FinancialCalculatorInput(BaseModel):
    """Entradas para la calculadora financiera."""
    P: float = Field(description="Principal amount")
    r: float = Field(description="Annual interest rate as decimal (e.g., 0.06 for 6%)")
    n: int = Field(description="Times compounded per year")
    t: float = Field(description="Time in years")

@tool("financial_calculator", args_schema=FinancialCalculatorInput)
def financial_calculator(P: float, r: float, n: int, t: float) -> str:
    """Computes compound interest and returns the result."""
    print(f"\n--- Ejecutando Herramienta (Langchain): financial_calculator(P={P}, r={r}, n={n}, t={t}) ---")
    result = P * (1 + r/n) ** (n*t)
    result_str = f"{result:,.2f}"
    print(f"--- Resultado Herramienta (Langchain): {result_str} ---")
    return result_str

# --- Función para parsear la llamada ---
def parse_tool_call(llm_output: str) -> dict | None:
    """Extrae el JSON de la ÚLTIMA <tool_call>."""
    print(f"\n--- Intentando parsear salida LLM (Turno 1):\n{llm_output}\n---")
    
    # --- CORRECCIÓN DE REGEX ---
    # Usamos .*(...) para encontrar la *última* coincidencia de <tool_call> en el
    # texto, evitando que se capture el ejemplo del prompt del sistema.
    match = re.search(r".*<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        # El JSON es el grupo 1 (la única captura)
        tool_call_json = match.group(1).strip()
        try:
            parsed = json.loads(tool_call_json)
            print(f"--- JSON parseado correctamente: {parsed} ---")
            return parsed
        except json.JSONDecodeError as e:
            print(f"Error: JSON malformado en <tool_call>: {e}")
            return None
    else:
        print("Advertencia: No se encontró <tool_call>.")
        return None

# --- Función para invocar la herramienta ---
def invoke_tool(parsed_call: dict | None) -> str:
    """Invoca la herramienta si el parseo fue exitoso."""
    if parsed_call and parsed_call.get("tool") == "financial_calculator" and "input" in parsed_call:
        return financial_calculator.invoke(parsed_call["input"])
    else:
        # Este error se enviará de vuelta al LLM en el Turno 2
        return "Error: No se pudo ejecutar la herramienta. La llamada (tool_call) no se generó o no se pudo parsear. Revisa tu salida."

# --- 2. Función Principal (main) ---
def main():
    model_id = "SUFE-AIFLM-Lab/Fin-R1"
    print(f"--- Iniciando experimento con langchain y modelo: {model_id} ---")

    # --- 3. Carga del Modelo y Tokenizer ---
    print("--- Cargando modelo y tokenizer (sin cuantización)... ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    print("--- Modelo y tokenizer cargados ---")

    # --- 4. Creación del Pipeline (Determinista) ---
    
    # --- LA SOLUCIÓN (APLICADA AQUÍ) ---
    # La pipeline 'tool_pipe' ahora incluye 'stop_sequences' como un
    # argumento de primer nivel. Esto forzará a model.generate() a
    # detenerse inmediatamente después de la etiqueta </tool_call>.
    
    # Pipeline para la llamada a la herramienta (salida corta)
    tool_pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        model_kwargs={"use_cache": True},
        max_new_tokens=150, 
        do_sample=False, 
        pad_token_id=model.config.pad_token_id,
        # --- ESTA ES LA CORRECCIÓN SOLICITADA ---
        stop_sequences=["</tool_call>"] 
    )
    tool_llm = HuggingFacePipeline(pipeline=tool_pipe)

    # Pipeline para la respuesta final (salida más larga)
    final_pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        model_kwargs={"use_cache": True},
        max_new_tokens=512, 
        do_sample=False, 
        pad_token_id=model.config.pad_token_id
        # No usamos 'stop_sequences' aquí
    )
    final_llm = HuggingFacePipeline(pipeline=final_pipe)
    print("--- Pipelines de Hugging Face creados ---")

    # --- 5. Creación de la Cadena LCEL ---
    # Prompt MUY explícito para generar SÓLO la llamada JSON
    tool_call_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that MUST use the financial_calculator tool.
Tool definition: financial_calculator(P: float, r: float, n: int, t: float).
You MUST output ONLY the tool call in JSON format wrapped in <tool_call> tags. DO NOT add any other text.
Example:
<tool_call>
{{"tool": "financial_calculator", "input": {{"P": 100, "r": 0.05, "n": 4, "t": 2}}}}
</tool_call>"""),
        ("user", "{input}")
    ])

    # Prompt para generar la respuesta final
    final_response_prompt = ChatPromptTemplate.from_messages([
         ("system", "You are a helpful financial assistant. You have received the result from a calculation."),
         ("user", "Original question: {original_input}\nCalculation Result: {tool_result}"),
         ("ai", "Based on the calculation, the answer to your question is:")
    ])

    # Cadena LCEL
    chain = (
        # Guardar la entrada original
        RunnablePassthrough.assign(original_input=lambda x: x["input"])
        # Generar la llamada
        | RunnablePassthrough.assign(llm_tool_call_output=tool_call_prompt | tool_llm | StrOutputParser())
        # Parsear la llamada (usando la regex mejorada)
        | RunnablePassthrough.assign(parsed_call=RunnableLambda(lambda x: parse_tool_call(x["llm_tool_call_output"])))
        # Ejecutar la herramienta
        | RunnablePassthrough.assign(tool_result=RunnableLambda(lambda x: invoke_tool(x["parsed_call"])))
        # Generar la respuesta final
        | final_response_prompt
        | final_llm
        | StrOutputParser()
    )
    print("--- Cadena LCEL creada ---")

    # --- 6. Ejecución del Query ---
    user_query = "I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years?"
    print(f"\n--- Query de Usuario: {user_query} ---")

    # Invocar la cadena completa
    final_answer = chain.invoke({"input": user_query})

    # Imprimir la respuesta final
    print(f"\n--- Respuesta Final del Agente: ---")
    print(final_answer)

if __name__ == "__main__":
    main()