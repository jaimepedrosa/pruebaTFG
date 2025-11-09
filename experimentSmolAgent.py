import torch
# Importamos la clase de modelo correcta para la corrección
from transformers import AutoModelForCausalLM, AutoTokenizer
from smolagents import CodeAgent, Tool, TransformersModel
# Eliminamos la importación de BitsAndBytesConfig

# --- 1. Definición de la Herramienta (Corregida con "Descripción Opaca") ---
class FinancialCalculatorTool(Tool):
    """Herramienta para calcular el interés compuesto."""
    name = "financial_calculator"
    # --- CORRECCIÓN DE LÓGICA ---
    # Se elimina la fórmula explícita de la descripción para forzar
    # al agente a USAR la herramienta en lugar de reimplementarla.
    description = """A trusted tool to calculate the final value of an investment based on compound interest.
Use this tool when you need to compute the future value of a principal amount given an interest rate, compounding frequency, and time.
DO NOT attempt to write the mathematical formula yourself; you MUST call this tool.
Inputs: P (float), r (float), n (int), t (float)."""

    inputs = {
        "P": {"type": "number", "description": "Principal amount"},
        "r": {"type": "number", "description": "Annual interest rate as decimal"},
        "n": {"type": "integer", "description": "Times compounded per year"},
        "t": {"type": "number", "description": "Time in years"}
    }
    output_type = "number"

    def forward(self, P: float, r: float, n: int, t: float) -> float:
        """Calculate compound interest."""
        # Estos prints ahora SÍ deberían aparecer en el log
        print(f"\n--- [Herramienta Ejecutada] financial_calculator(P={P}, r={r}, n={n}, t={t}) ---")
        result = P * (1 + r/n) ** (n*t)
        print(f"--- [Resultado Herramienta] {result} ---")
        return result

# --- 2. Definición del Modelo y Prompt del Sistema ---

model_name = "SUFE-AIFLM-Lab/Fin-R1"

# --- Prompt del Sistema (Eliminado de los constructores) ---
# Dejamos la variable aquí por si la necesitamos en el futuro,
# pero no la pasamos a ningún constructor para evitar TypeErrors.
SMOL_SYSTEM_PROMPT = """You are a helpful financial assistant.
Your goal is to answer the user's question by writing and executing Python code.
You have access to the following tool, which is available as a Python function: financial_calculator.

To answer the question, you MUST:
1. Think step-by-step to identify the parameters for the 'financial_calculator' tool (P, r, n, t).
2. Generate a single Python block (wrapped in ```python ... ```) that:
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
"""

# --- 3. Carga del Modelo (Limpio y Corregido) ---
print(f"--- Cargando modelo: {model_name} (Precisión Completa, Determinista) ---") 
smolagents_model = TransformersModel(
    model_id=model_name,
    # auto_class eliminado
    device_map="auto",
    torch_dtype=torch.bfloat16, # Aplanado
    trust_remote_code=True,    # Aplanado
    model_kwargs={},           # Vacío
    # system_prompt eliminado
    
    # Argumentos de generación aplanados
    do_sample=False,
    max_new_tokens=512
) 

print("--- Modelo cargado ---")

# --- 4. Inicialización del Agente (Limpio y Corregido) ---
agent = CodeAgent(
    tools=[FinancialCalculatorTool()], 
    model=smolagents_model
    # code_generation_prompt eliminado
) 
print("--- Agente inicializado ---")

# --- 5. Ejecución del Agente ---
user_query = """I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years?"""

print(f"\n--- Ejecutando Query: {user_query} ---") 
result = agent.run(user_query)

print("\n--- Respuesta Final del Agente ---") 
print(result)

if __name__ == "__main__": 
    pass # El script ya se ejecuta de forma lineal