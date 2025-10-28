import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
# from peft import PeftModel  <--- YA NO ES NECESARIO
import sys

# --- CLASE DE PARADA PERSONALIZADA ---
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
# ----------------------------------------

# --- PASO 1: DEFINIR MODELOS Y CONFIGURACIÓN ---
model_name = "SUFE-AIFLM-Lab/Fin-R1"
# adapter_path = "./finr1-qlora-adapter"  <--- YA NO ES NECESARIO

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
model = AutoModelForCausalLM.from_pretrained(  # <-- Este es ahora el modelo final
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

# --- Lista de Criterios de Parada ---
stop_token_ids = [
    tokenizer.bos_token_id, 
    tokenizer.encode("[INST]", add_special_tokens=False)[-1] 
]
stopping_criteria_list = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
# -------------------------------------

# --- PASO 4: FUSIONAR EL ADAPTADOR LoRA CON EL MODELO ---
# print(f"Cargando y fusionando el adaptador desde: {adapter_path}")
# model = PeftModel.from_pretrained(model, adapter_path) <--- SECCIÓN ELIMINADA
model.eval() # Ponemos el modelo base en modo de evaluación

print("\n--- ✅ Modelo BASE listo para chatear ---")
print('Escribe tu prompt. Escribe "salir" para terminar.\n')

# --- PASO 5: BUCLE DE CONVERSACIÓN ---
try:
    while True:
        prompt = input("Tú: ")
        if prompt.lower() == "salir":
            break
            
        formatted_prompt = f"<s>[INST] {prompt} [/INST] "
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        prompt_length = inputs["input_ids"].shape[1]

        print("FinR1 (Base): (Generando respuesta...)")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, 
                do_sample=True,          
                temperature=0.6,
                top_p=0.9,               
                stopping_criteria=stopping_criteria_list 
            )
        
        new_tokens = outputs[0][prompt_length:]
        model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"FinR1 (Base): {model_answer}\n")

except KeyboardInterrupt:
    print("\nSaliendo de la demo.")
    sys.exit()

print("\n¡Hasta luego!")