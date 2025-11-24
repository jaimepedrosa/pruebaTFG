import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# --- 1. Constantes y Configuración ---
# Basado en el análisis del paper (Plan Experimental)

# Modelo: SUFE-AIFLM-Lab/Fin-R1
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"

# Dataset: El paper identifica FinCorpus (Duxiaoman-DI/FinCorpus)
DATASET_ID = "Duxiaoman-DI/FinCorpus"
DATASET_SPLIT = "test" # Usamos el split de evaluación

# Archivo de salida para la evaluación "LLM-como-Juez"
OUTPUT_FILE = "baseline_outputs.jsonl"

# Parámetros de Inferencia (¡CRÍTICO!):
# Replicando la metodología del paper (Sección 2.2.2) que usa muestreo.
INFERENCE_TEMP = 0.6
DO_SAMPLE = True
MAX_NEW_TOKENS = 1024 # Un límite razonable para las respuestas

# Sistema de prompt (Estándar para el modelo base Qwen2.5)
SYSTEM_PROMPT = "You are a helpful AI Assistant."

def main():
    print(f"--- Iniciando Inferencia Baseline ---")
    print(f"Modelo: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID} (Split: {DATASET_SPLIT})")
    
    # --- 2. Cargar Tokenizer y Modelo ---
    print("Cargando tokenizer...")
    # El tokenizer usa 'trust_remote_code=True' como es estándar para Qwen
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    
    # Asegurar que el token de padding esté configurado para la generación
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo (torch.bfloat16, device_map='auto')...")
    # Carga del modelo sin cuantización, como se especificó
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Establecer el modelo en modo de evaluación (desactiva dropout, etc.)
    model.eval()
    print(f"Modelo cargado en el dispositivo: {model.device}")

    # --- 3. Cargar Dataset ---
    print(f"Cargando dataset {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)
    
    # Seleccionar solo el split de test
    try:
        test_dataset = dataset[DATASET_SPLIT]
    except KeyError:
        print(f"Error: El split '{DATASET_SPLIT}' no se encontró.")
        print(f"Splits disponibles: {list(dataset.keys())}")
        return

    print(f"Dataset cargado. Número de muestras en '{DATASET_SPLIT}': {len(test_dataset)}")

    # --- 4. Iterar, Generar y Guardar Resultados ---
    print(f"Iniciando generación... Los resultados se guardarán en {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # Usamos tqdm para una barra de progreso
        for sample in tqdm(test_dataset, desc="Procesando FinCorpus (test)"):
            
            # Extraer datos de la muestra
            # (Basado en la estructura de 'Duxiaoman-DI/FinCorpus': id, input, output)
            sample_id = sample.get('id', 'N/A')
            prompt_text = sample.get('input')
            ground_truth = sample.get('output')

            if not prompt_text:
                print(f"Saltando muestra {sample_id} por 'input' vacío.")
                continue

            # Formatear el prompt usando la plantilla de chat del modelo
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ]
            
            try:
                # Aplicar plantilla y tokenizar
                prompt_chat_templated = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(
                    prompt_chat_templated,
                    return_tensors="pt"
                ).to(model.device)

                # Generar la respuesta usando los parámetros del paper
                outputs = model.generate(
                    **inputs,
                    do_sample=DO_SAMPLE,
                    temperature=INFERENCE_TEMP,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decodificar solo los tokens generados (nuevos)
                # Se resta la longitud del input para obtener solo la respuesta
                response_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response_text = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True
                ).strip()

                # Crear el objeto de resultado
                result_data = {
                    "id": sample_id,
                    "pregunta": prompt_text,
                    "respuesta_ground_truth": ground_truth,
                    "respuesta_baseline": response_text
                }
                
                # Escribir el resultado como una línea JSON
                f_out.write(json.dumps(result_data, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"\nError procesando la muestra {sample_id}: {e}")
                # Guardar un error si falla la generación
                error_data = {
                    "id": sample_id,
                    "pregunta": prompt_text,
                    "respuesta_ground_truth": ground_truth,
                    "respuesta_baseline": f"ERROR_GENERACION: {str(e)}"
                }
                f_out.write(json.dumps(error_data, ensure_ascii=False) + '\n')

    print(f"--- Inferencia Baseline Completada ---")
    print(f"Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()