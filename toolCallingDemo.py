import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

print("--- 0. Cargando el modelo y el tokenizador (SIN CUANTIZACIÓN) ---")

# 1: MODELO SIN CUANTIZACIÓN ---
# Modelo Fin-R1, cargado en bfloat16 para máxima precisión, en lugar de usar 8-bit o 4-bit (QLoRA).
model_name = "SUFE-AIFLM-Lab/Fin-R1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Carga en bfloat16 (precisión original)
    device_map="auto",           # Distribuye el modelo en las GPUs disponibles
    trust_remote_code=True       # Necesario para este modelo
)

# Como los modelos Llama (base de Fin-R1) no suelen tener pad_token, lo configuramos al eos_token para evitar warnings durante la generación
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n--- Modelo cargado exitosamente ---")

# --- 2: IMPLEMENTACIÓN DE TOOL-CALLING ---

# 1. Definir la Herramienta (System Prompt): Se define la herramienta y el formato de salida JSON exacto.

system_prompt = """You are an assistant that can use tools when helpful.

Available tool:
- financial_calculator(P: float, r: float, n: int, t: float): compute compound interest using A = P * (1 + r/n)^(n*t)

When you decide to use a tool, output ONLY a JSON object wrapped between <tool_call> and </tool_call> tags:
<tool_call>
{"tool": "financial_calculator", "input": {"P": <principal>, "r": <annual_rate>, "n": <times_compounded_per_year>, "t": <years>}}
</tool_call>
"""

# Pregunta
user_prompt = """I have $10,000 to invest. The bank offers an interest rate of 6% per year, compounded monthly (12 times per year).
How much money will I have after 10 years? Use the tool to find the exact result."""

# 2. Turno 1: Generar la llamada a la herramienta
print("\n--- 1. Generando llamada a la herramienta (Turno 1) ---")

messages_turn_1 = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# El tokenizador de Fin-R1 (Llama 2) formateará esto en el prompt <s>[INST] <<SYS>>...<</SYS>>... [/INST]
prompt_turn_1 = tokenizer.apply_chat_template(
    messages_turn_1,
    tokenize=False,
    add_generation_prompt=True
)

inputs_turn_1 = tokenizer(prompt_turn_1, return_tensors="pt").to(model.device)

with torch.no_grad():
    # --- DIRECTRIZ 2: GENERACIÓN DETERMINISTA ---
    # Usamos do_sample=False (greedy decoding) sin temperature/top_p para asegurar que la salida sea siempre la misma.
    outputs_turn_1 = model.generate(
        **inputs_turn_1,
        max_new_tokens=128,          # Suficiente para el JSON
        do_sample=False,             # Obligatorio para determinismo
        pad_token_id=tokenizer.eos_token_id
    )

# Decodificamos solo la parte generada (excluyendo el prompt de entrada)
tool_call_output = tokenizer.decode(
    outputs_turn_1[0][inputs_turn_1["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print(f"Salida del modelo (raw): {tool_call_output}")

# 3. Parseo y Ejecución de la Herramienta
print("\n--- 2. Parseando y ejecutando la herramienta ---")

# Extraemos el JSON de las etiquetas <tool_call>
match = re.search(r"<tool_call>(.*?)</tool_call>", tool_call_output, re.DOTALL | re.IGNORECASE)

if not match:
    raise ValueError("Error: El modelo no generó una llamada a la herramienta válida.")

tool_call_json = match.group(1).strip()
tool_data = json.loads(tool_call_json)

print(f"JSON de la herramienta extraído: {tool_data}")

# Definimos la función de Python que SÍ HACE el cálculo
def financial_calculator(P: float, r: float, n: int, t: float) -> float:
    """Calcula el interés compuesto."""
    return P * (1 + r/n) ** (n*t)

# Ejecutamos la función con los argumentos del JSON
tool_result = financial_calculator(**tool_data["input"])
tool_result_str = f"{tool_result:,.2f}" # Formateado como string

print(f"Resultado de la herramienta (cálculo real): {tool_result_str}")

# 4. Turno 2: Generar la Respuesta Final
print("\n--- 3. Generando respuesta final (Turno 2) ---")

# Construimos el nuevo historial de chat, incluyendo la llamada del modelo
# y la respuesta de nuestra herramienta.
messages_turn_2 = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    {
        "role": "assistant",
        "content": f"<tool_call>{tool_call_json}</tool_call>" # Lo que el modelo dijo
    },
    {
        "role": "tool",
        "name": "financial_calculator",
        "content": tool_result_str  # El resultado de la función
    }
]

prompt_turn_2 = tokenizer.apply_chat_template(
    messages_turn_2,
    tokenize=False,
    add_generation_prompt=True # Añade el token [/INST] para que el modelo responda
)

inputs_turn_2 = tokenizer(prompt_turn_2, return_tensors="pt").to(model.device)

with torch.no_grad():
    # --- DIRECTRIZ 2: GENERACIÓN DETERMINISTA (de nuevo) ---
    outputs_turn_2 = model.generate(
        **inputs_turn_2,
        max_new_tokens=256,         # Más espacio para una respuesta en lenguaje natural
        do_sample=False,            # Obligatorio para determinismo
        pad_token_id=tokenizer.eos_token_id
    )

# Decodificamos la respuesta final
final_answer = tokenizer.decode(
    outputs_turn_2[0][inputs_turn_2["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print(f"\n--- RESPUESTA FINAL (del Modelo) --- \n{final_answer}")