import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,        # <--- CAMBIO: Importación necesaria
    StoppingCriteriaList     # <--- CAMBIO: Importación necesaria
)
from peft import PeftModel
import sys

# --- CAMBIO: Definir una clase de parada personalizada ---
# Esta clase comprobará cada token nuevo que se genere.
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Comprueba si el ÚLTIMO token generado es uno de los tokens de parada
        if input_ids[0][-1] in self.stop_token_ids:
            return True
        return False
# ----------------------------------------------------

# --- PASO 1: DEFINIR MODELOS Y CONFIGURACIÓN ---
model_name = "SUFE-AIFLM-Lab/Fin-R1"
adapter_path = "./finr1-qlora-adapter"

print("Iniciando la carga del modelo...")

# --- PASO 2: CONFIGURACIÓN DE QUANTIZACIÓN ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- PASO 3: CARGAR MODELO BASE Y TOKENIZER ---
print(f"Cargando el modelo base (desde caché local): {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

print(f"Cargando el tokenizer (desde caché local)...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

# --- CAMBIO: Crear la lista de criterios de parada ---
# Le diremos que se detenga si ve el token de inicio (bos) o el token [INST]
stop_token_ids = [
    tokenizer.bos_token_id, # ID del token <s>
    tokenizer.encode("[INST]", add_special_tokens=False)[-1] # ID del token [INST]
]
stopping_criteria_list = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
# --------------------------------------------------

# --- PASO 4: FUSIONAR EL ADAPTADOR LoRA CON EL MODELO ---
print(f"Cargando y fusionando el adaptador desde: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print("\n--- ✅ Modelo listo para chatear ---")
print('Escribe tu prompt. Escribe "salir" para terminar.\n')

# --- PASO 5: BUCLE DE CONVERSACIÓN ---
try:
    while True:
        prompt = input("Tú: ")
        if prompt.lower() == "salir":
            break
            
        formatted_prompt = f"<s>[INST] {prompt} [/INST] "
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        print("FinR1: (Generando respuesta...)")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, 
                # --- CAMBIOS EN LA GENERACIÓN ---
                do_sample=True,          # Volvemos a True para evitar bucles
                temperature=0.6,         # Un poco menos "loco"
                top_p=0.9,               
                stopping_criteria=stopping_criteria_list # Aplicamos la regla de parada
                # ----------------------------------
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = full_response.split("[/INST]")[-1].strip()
        
        # --- CAMBIO: Limpiar la salida de tokens basura ---
        model_answer = model_answer.replace("[OUT]", "").replace("[/S]", "").replace("```python", "")
        # -------------------------------------------------
        
        print(f"FinR1: {model_answer}\n")

except KeyboardInterrupt:
    print("\nSaliendo de la demo.")
    sys.exit()

print("\n¡Hasta luego!")