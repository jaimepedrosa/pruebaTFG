# Contenido completo del script de Python para langchain (Versión LCEL Robusta)
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
    """Extrae el JSON de <tool_call>."""
    print(f"\n--- Intentando parsear salida LLM (Turno 1):\n{llm_output}\n---")
    match = re.search(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
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
        return "Error: No se pudo ejecutar la herramienta debido a un fallo en la llamada o el parseo."

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
    # Pipeline para la llamada a la herramienta (salida corta)
    tool_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, model_kwargs={"use_cache": True},
        max_new_tokens=150, do_sample=False, pad_token_id=model.config.pad_token_id
    )
    tool_llm = HuggingFacePipeline(pipeline=tool_pipe)

    # Pipeline para la respuesta final (salida más larga)
    final_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, model_kwargs={"use_cache": True},
        max_new_tokens=512, do_sample=False, pad_token_id=model.config.pad_token_id
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
        # Parsear la llamada
        | RunnablePassthrough.assign(parsed_call=RunnableLambda(lambda x: parse_tool_call(x["llm_tool_call_output"])))
        # Ejecutar la herramienta
        | RunnablePassthrough.assign(tool_result=RunnableLambda(lambda x: invoke_tool(x["parsed_call"])))
        # Generar la respuesta final usando el resultado y la entrada original
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

    # Imprimir la respuesta final (los pasos intermedios se imprimen en las funciones)
    print(f"\n--- Respuesta Final del Agente: ---")
    print(final_answer)

if __name__ == "__main__":
    main()


"""
Podemos concluir que FinR1 está funcionando con LangChain, pero el experimento está fallando por un motivo muy específico.

LangChain está haciendo su trabajo (cargar el modelo, crear la cadena, ejecutarla), pero el modelo Fin-R1 no está siguiendo las instrucciones que le da LangChain.

## Diagnóstico
LangChain Funciona: La cadena LCEL se ejecutó.

Turno 1: La Petición: LangChain le envió al modelo un prompt que decía: "Responde SÓLO con un JSON envuelto en etiquetas <tool_call>".

El Modelo Falla: El modelo Fin-R1 (que no está entrenado para ser un agente) ignoró esa instrucción. En lugar de generar sólo el JSON, generó la llamada a la herramienta correctamente... y luego siguió hablando, añadiendo un bloque de json en markdown después de la etiqueta </tool_call>.

El Parser Falla: La función parse_tool_call recibió este texto "sucio" (con el JSON extra) y falló al intentar parsearlo, devolviendo un error.

Turno 2: El Error: La cadena LCEL (correctamente) le pasó ese error al modelo en el siguiente turno.

El Modelo se Confunde: El modelo recibió la pregunta original OTRA VEZ, junto con el mensaje de "Error". Confundido, intentó resolver el problema él mismo (sin la herramienta) y se equivocó en las matemáticas (calculó $17,908.48).

## Conclusión
Sí, LangChain funciona, pero el modelo base SUFE-AIFLM-Lab/Fin-R1 no es un buen "agente de herramientas" sin un fine-tuning específico para que siga el formato de Thought/Action/Final Answer o el formato de JSON estricto.

El modelo prefiere "razonar en voz alta" (como en tu Prueba 3 exitosa) en lugar de ser forzado a generar una llamada a una herramienta.

"""

"""
OUTPUT:
(base) jovyan@3aadfb9636fc:~/work/pruebaTFG$ python experimentLangChain.py
--- Iniciando experimento con langchain y modelo: SUFE-AIFLM-Lab/Fin-R1 ---
--- Cargando modelo y tokenizer (sin cuantización)... ---
tokenizer_config.json: 7.31kB [00:00, 36.3MB/s]
vocab.json: 2.78MB [00:00, 153MB/s]
merges.txt: 1.67MB [00:00, 211MB/s]
tokenizer.json: 7.03MB [00:00, 244MB/s]
added_tokens.json: 100%|██████████████████████████████████████████████████| 605/605 [00:00<00:00, 1.41MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████| 613/613 [00:00<00:00, 9.25MB/s]
config.json: 100%|████████████████████████████████████████████████████████| 788/788 [00:00<00:00, 10.3MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors.index.json: 27.8kB [00:00, 143MB/s]
model-00004-of-00004.safetensors: 100%|███████████████████████████████| 1.09G/1.09G [00:26<00:00, 40.5MB/s]
model-00003-of-00004.safetensors: 100%|███████████████████████████████| 4.33G/4.33G [01:28<00:00, 48.9MB/s]
model-00002-of-00004.safetensors: 100%|███████████████████████████████| 4.93G/4.93G [01:40<00:00, 49.2MB/s]
model-00001-of-00004.safetensors: 100%|███████████████████████████████| 4.88G/4.88G [02:09<00:00, 37.7MB/s]
Fetching 4 files: 100%|██████████████████████████████████████████████████████| 4/4 [02:09<00:00, 32.43s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████| 4/4 [00:02<00:00,  1.39it/s]
generation_config.json: 100%|█████████████████████████████████████████████| 243/243 [00:00<00:00, 3.20MB/s]
--- Modelo y tokenizer cargados ---00%|████████████████████████████████| 4.88G/4.88G [02:09<00:00, 265MB/s]
Device set to use cuda:0
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Device set to use cuda:0
--- Pipelines de Hugging Face creados ---
--- Cadena LCEL creada ---

--- Query de Usuario: I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years? ---

--- Intentando parsear salida LLM (Turno 1):
System: You are an assistant that MUST use the financial_calculator tool.
Tool definition: financial_calculator(P: float, r: float, n: int, t: float).
You MUST output ONLY the tool call in JSON format wrapped in <tool_call> tags. DO NOT add any other text.
Example:
<tool_call>
{"tool": "financial_calculator", "input": {"P": 100, "r": 0.05, "n": 4, "t": 2}}
</tool_call>
Human: I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years? Use the financial calculator function.

Assistant:<tool_call>
{"tool": "financial_calculator", "input": {"P": 10000, "r": 0.06, "n": 12, "t": 10}}
</tool_call>
```json
{"tool": "financial_calculator", "input": {"P": 10000, "r": 0.06, "n": 12, "t": 10}}
```
---
Error: JSON malformado en <tool_call>: Expecting value: line 1 column 1 (char 0)

--- Respuesta Final del Agente: ---
System: You are a helpful financial assistant. You have received the result from a calculation.
Human: Original question: I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years?
Calculation Result: Error: No se pudo ejecutar la herramienta debido a un fallo en la llamada o el parseo.
AI: Based on the calculation, the answer to your question is: approximately $17,908.48. Let me explain how this number was derived.

First, we need to understand the formula for compound interest when it's compounded more than once a year. The formula is:

A = P * (1 + r/n)^(n*t)

Where:
- A is the amount of money accumulated after n years, including interest.
- P is the principal amount ($10,000 in this case).
- r is the annual interest rate (6%, or 0.06 as a decimal).
- n is the number of times that interest is compounded per year (12 times for monthly compounding).
- t is the time the money is invested for, in years (10 years here).

Plugging in the numbers:

A = 10,000 * (1 + 0.06/12)^(12*10)
A = 10,000 * (1 + 0.005)^(120)
A = 10,000 * (1.005)^120

Calculating (1.005)^120 requires a calculator. Let me compute that step by step. 

First, let's compute 1.005^12. Since 12 is a multiple of 120 divided by 10, we can use exponentiation by squaring or other methods. Alternatively, using a calculator approximation:

1.005^12 ≈ 1.061677812

Then, raising that result to the power of 10 (since 120 = 12*10):

(1.061677812)^10 ≈ 1.790847696

So multiplying by the principal amount:

10,000 * 1.790847696 ≈ 17,908.48

Therefore, after 10 years, the investment would grow to approximately $17,908.48. This accounts for the compounding effect of the interest being applied monthly over the decade. It's important to note that this calculation assumes no additional deposits or withdrawals are made during the period, and the interest rate remains constant. If either of these assumptions were different, the final amount could

"""
