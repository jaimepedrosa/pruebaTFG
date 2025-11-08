import torch
# Importamos la clase de modelo correcta para la corrección
from transformers import AutoModelForCausalLM, AutoTokenizer
from smolagents import CodeAgent, Tool, TransformersModel
# Eliminamos la importación de BitsAndBytesConfig

# --- 1. Definición de la Herramienta (Sin cambios) ---
class FinancialCalculatorTool(Tool):
    """Herramienta para calcular el interés compuesto."""
    name = "financial_calculator"
    description = """Computes compound interest using the formula A = P * (1 + r/n)^(n*t) where:
    - P is the principal amount
    - r is the annual interest rate as a decimal (e.g., 0.06 for 6%)
    - n is the number of times interest is compounded per year
    - t is the time in years"""

    inputs = {
        "P": {"type": "number", "description": "Principal amount"},
        "r": {"type": "number", "description": "Annual interest rate as decimal"},
        "n": {"type": "integer", "description": "Times compounded per year"},
        "t": {"type": "number", "description": "Time in years"}
    }
    output_type = "number"

    def forward(self, P: float, r: float, n: int, t: float) -> float:
        """Calculate compound interest."""
        print(f"\n--- [Herramienta Ejecutada] financial_calculator(P={P}, r={r}, n={n}, t={t}) ---")
        result = P * (1 + r/n) ** (n*t)
        print(f"--- [Resultado Herramienta] {result} ---")
        return result

# --- 2. Definición del Modelo y Prompt del Sistema ---

model_name = "SUFE-AIFLM-Lab/Fin-R1"

# --- SOLUCIÓN: Prompt del Sistema Corregido ---
# El bloque de código de ejemplo (```python ... ```) ahora está
# correctamente cerrado con ``` y el print() está dentro de él.
# El string SMOL_SYSTEM_PROMPT se cierra al final con """.
SMOL_SYSTEM_PROMPT = """You are a helpful financial assistant.
Your goal is to answer the user's question by writing and executing Python code.
You have access to the following tool, which is available as a Python function: financial_calculator.

To answer the question, you MUST:
1. Think step-by-step to identify the parameters for the 'financial_calculator' tool (P, r, n, t).
2. Generate a single Python code block (wrapped in ```python ... ```) that:
   - Imports the tool: `from tools import financial_calculator`
   - Sets the parameters from the user's query.
   - Calls the tool with those parameters.
   - Prints the final calculated result.

Example of a perfect response:

Thought: The user needs to calculate compound interest.
P is $10,000. r is 6% (0.06). n is 12 (monthly). t is 10 years.
I will write a Python block to call `financial_calculator(P=10000, r=0.06, n=12, t=10)` and print the output.
```python
from tools import financial_calculator

P = 10000.0
r = 0.06
n = 12
t = 10.0
result = financial_calculator(P=P, r=r, n=n, t=t)
print(f"The total amount after 10 years will be: ${result:,.2f}")
```
"""

# --- 3. Carga del Modelo (Corregida) ---
print(f"--- Cargando modelo: {model_name} (Precisión Completa, Determinista) ---")
smolagents_model = TransformersModel(
    model_id=model_name,
    # Forzamos la clase AutoModelForCausalLM para evitar el error de 'torch_dtype'
    auto_class=AutoModelForCausalLM, 
    device_map="auto",
    model_kwargs={
        # SOLUCIÓN 1: Sin cuantización, usamos bfloat16
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True
        # Eliminada "quantization_config"
    },
    generation_kwargs={
        # SOLUCIÓN 2: Generación determinista
        "do_sample": False,
        "max_new_tokens": 512 # Espacio para el "Thought" y el código
    }
)
print("--- Modelo cargado ---")

# --- 4. Inicialización del Agente (Corregida) ---
agent = CodeAgent(
    tools=[FinancialCalculatorTool()],
    model=smolagents_model,
    system_prompt=SMOL_SYSTEM_PROMPT # SOLUCIÓN 3: Inyectamos el prompt
)
print("--- Agente inicializado ---")

# --- 5. Ejecución del Agente ---
user_query = """I have $10,000 to invest. The bank offers an interest rate of 6% per year,
compounded monthly (12 times per year). How much money will I have after 10 years?"""

print(f"\n--- Ejecutando Query: {user_query} ---")
# Esta llamada ya no debería colgarse
result = agent.run(user_query)

print("\n--- Respuesta Final del Agente ---")
print(result)

if __name__ == "__main__":
    pass # El script ya se ejecuta de forma lineal