
# --- IMPORTACIONES ---
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
import sys

# --- CLASE DE PARADA PERSONALIZADA ---
# Se detiene si el modelo intenta generar un nuevo prompt de usuario
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            # Comprueba si el ÚLTIMO token es un token de parada
            if input_ids[0][-1] == stop_id:
                return True
        return False
# ----------------------------------------

# --- PASO 1: CARGAR MODELO Y TOKENIZER ---
print("--- 0. Cargando el modelo y el tokenizador (SIN CUANTIZACIÓN) ---")

model_name = "SUFE-AIFLM-Lab/Fin-R1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Carga en bfloat16 (precisión original)
    device_map="auto",           # Distribuye el modelo en las GPUs disponibles
    trust_remote_code=True       # Necesario para este modelo
)

# Configurar pad_token para evitar warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Lista de Criterios de Parada ---
# Detenerse si ve el token de inicio (<s>) o el token [INST]
stop_token_ids = [
    tokenizer.bos_token_id, 
    tokenizer.encode("[INST]", add_special_tokens=False)[-1] 
]
stopping_criteria_list = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
# -------------------------------------

model.eval() # Poner el modelo en modo de evaluación

print("\n--- ✅ Modelo listo para chatear (Modo Determinista) ---")
print('Escribe tu prompt. Escribe "salir" para terminar.\n')

# --- PASO 2: BUCLE DE CHAT ---
try:
    while True:
        prompt = input("Tú: ")
        if prompt.lower() == "salir":
            break
            
        # Formatear el prompt usando la plantilla de chat de Llama 2
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # El tokenizador de Fin-R1 (Llama 2) formateará esto
        # en el prompt <s>[INST] ... [/INST]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Guardamos la longitud del prompt para poder separarlo de la respuesta
        prompt_length = inputs["input_ids"].shape[1]

        print("FinR1: (Generando respuesta...)")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,                  # Límite de la respuesta
                do_sample=False,                     # --- GENERACIÓN DETERMINISTA ---
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, # Parar al final de la sentencia
                stopping_criteria=stopping_criteria_list # Parar si alucina "[INST]"
            )
        
        # --- DECODIFICACIÓN LIMPIA ---
        # 1. Obtenemos solo los tokens generados (excluyendo el prompt)
        new_tokens = outputs[0][prompt_length:]
        # 2. Decodificamos solo esos nuevos tokens
        model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # ------------------------------------------------
        
        print(f"FinR1: {model_answer}\n")

except KeyboardInterrupt:
    print("\nSaliendo de la demo.")
    sys.exit()

print("\n¡Hasta luego!")