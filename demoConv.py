import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import sys

# --- PASO 1: DEFINIR MODELOS Y CONFIGURACIÓN ---

# El modelo base original (el mismo que en el entrenamiento)
model_name = "SUFE-AIFLM-Lab/Fin-R1"

# La ruta a tu adaptador LoRA (la carpeta que se guardó al entrenar)
adapter_path = "./finr1-qlora-adapter"

print("Iniciando la carga del modelo...")

# --- PASO 2: CONFIGURACIÓN DE QUANTIZACIÓN ---
# DEBE ser exactamente la misma que usaste para entrenar.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- PASO 3: CARGAR MODELO BASE Y TOKENIZER ---
print(f"Cargando el modelo base: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print(f"Cargando el tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- PASO 4: FUSIONAR EL ADAPTADOR LoRA CON EL MODELO ---
# Esta es la parte clave: cargamos el adaptador y lo fusionamos
# con el modelo base que acabamos de cargar.
print(f"Cargando y fusionando el adaptador desde: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

# Ponemos el modelo en modo de evaluación (inferencia)
model.eval()

print("\n--- ✅ Modelo listo para chatear ---")
print('Escribe tu prompt. Escribe "salir" para terminar.\n')

# --- PASO 5: BUCLE DE CONVERSACIÓN ---
try:
    while True:
        # 1. Pedir input al usuario
        prompt = input("Tú: ")
        if prompt.lower() == "salir":
            break
            
        # 2. Formatear el prompt como lo espera el modelo
        # Este formato (Llama 2 Chat) es VITAL.
        formatted_prompt = f"<s>[INST] {prompt} [/INST] "

        # 3. Tokenizar el input
        # No es necesario el .to("cuda") gracias a device_map="auto"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        # 4. Generar la respuesta
        print("FinR1: (Generando respuesta...)")
        with torch.no_grad(): # Desactiva el cálculo de gradientes para ahorrar memoria
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=250,      # Número máximo de tokens a generar
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,          # Activa el muestreo para respuestas más creativas
                temperature=0.7,         # Controla la aleatoriedad (más bajo = más determinista)
                top_p=0.9                # Muestreo Nucleus
            )
        
        # 5. Decodificar y mostrar solo la respuesta
        # La salida completa incluye tu prompt, así que lo extraemos.
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = full_response.split("[/INST]")[-1].strip()
        
        print(f"FinR1: {model_answer}\n")

except KeyboardInterrupt:
    print("\nSaliendo de la demo.")
    sys.exit()

print("\n¡Hasta luego!")