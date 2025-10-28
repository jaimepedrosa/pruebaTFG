import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import sys

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
    local_files_only=True  # <--- ARREGLO 1: Evita descargar el modelo cada vez
)

print(f"Cargando el tokenizer (desde caché local)...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    local_files_only=True  # <--- ARREGLO 1: Evita descargar el tokenizer cada vez
)
tokenizer.pad_token = tokenizer.eos_token

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

        # --- ARREGLO 2: Mueve los inputs a la GPU (model.device) ---
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        print("FinR1: (Generando respuesta...)")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,                  # <--- ARREGLO 2: Pasa los inputs de la GPU
                max_new_tokens=512,        # Aumentado por si la respuesta es larga
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, # <--- ARREGLO 3: Sabe cuándo parar
                do_sample=True,          
                temperature=0.7,         
                top_p=0.9                
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = full_response.split("[/INST]")[-1].strip()
        
        print(f"FinR1: {model_answer}\n")

except KeyboardInterrupt:
    print("\nSaliendo de la demo.")
    sys.exit()

print("\n¡Hasta luego!")