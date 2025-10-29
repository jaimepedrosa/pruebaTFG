# Contenido completo del script de Python para smolagents
import torch
# --- CAMBIO: Importar la clase de modelo correcta ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from smolagents import CodeAgent, Tool, TransformersModel

# --- 1. Definición de la Herramienta (Estilo smolagents) ---
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
        """Calcula el interés compuesto."""
        print(f"\n--- Ejecutando Herramienta: financial_calculator(P={P}, r={r}, n={n}, t={t}) ---")
        result = P * (1 + r/n) ** (n*t)
        print(f"--- Resultado Herramienta: {result} ---")
        return result

# --- 2. Función Principal (main) ---
def main():
    model_id = "SUFE-AIFLM-Lab/Fin-R1"
    print(f"--- Iniciando experimento con smolagents y modelo: {model_id} ---")

    # --- 3. Carga del Modelo (TransformersModel) ---
    print("--- Cargando modelo (sin cuantización)... ---")
    smolagents_model = TransformersModel(
        model_id=model_id,
        # --- CAMBIO: Forzar la clase de modelo correcta ---
        # Esto evita que smolagents intente cargarlo como un modelo de Imagen-a-Texto
        auto_class=AutoModelForCausalLM,
        # ------------------------------------------------
        device_map="auto",
        model_kwargs={
            "torch_dtype": torch.bfloat16, # Carga en precisión original
            "trust_remote_code": True      # Requerido por Fin-R1
        },
        generation_kwargs={              # Parámetros para inferencia determinista
            "do_sample": False,
            "max_new_tokens": 512,
        }
    )
    print("--- Modelo cargado ---")

    # --- 4. Inicialización del Agente ---
    agent = CodeAgent(
        tools=[FinancialCalculatorTool()],
        model=smolagents_model,
        system_prompt="You are a helpful assistant that uses tools. Respond with the final answer based on the tool result."
    )
    print("--- Agente smolagents inicializado ---")

    # --- 5. Ejecución del Query ---
    user_query = "I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year). How much money will I have after 10 years? Use the tool."
    
    print(f"\n--- Query de Usuario: {user_query} ---")
    result = agent.run(user_query)
    print(f"\n--- Respuesta Final del Agente: {result} ---")

if __name__ == "__main__":
    main()