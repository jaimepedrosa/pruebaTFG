import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm

# --- 1. Constantes y Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"

# Usamos FinEval (benchmark de los autores) como proxy de Ant-Finance
DATASET_ID = "SUFE-AIFLM-Lab/FinEval"

# ¡CRÍTICO PARA TU TFG!
# Seleccionamos subtemas de "Compliance" y "Negocio" donde el Agente debería brillar.
# 'law': Preguntas legales/regulatorias.
# 'banking': Preguntas de operativa bancaria.
TARGET_SUBSETS = ['law', 'banking'] 

# Usamos 'val' porque 'test' no tiene etiquetas públicas (ground truth)
DATASET_SPLIT = "val"

OUTPUT_FILE = "baseline_outputs_fineval.jsonl"

# Parámetros del Paper (Replicación exacta)
INFERENCE_TEMP = 0.6
DO_SAMPLE = True
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = "You are a helpful financial expert AI."

def format_fineval_prompt(sample):
    """
    Convierte una fila de FinEval (pregunta + opciones) en un string claro.
    """
    question = sample['question']
    options = f"A. {sample['A']}\nB. {sample['B']}\nC. {sample['C']}\nD. {sample['D']}"
    
    # Prompt diseñado para forzar razonamiento antes de la respuesta
    return f"Question:\n{question}\n\nOptions:\n{options}\n\nPlease analyze the question and select the correct option."

def main():
    print(f"--- Iniciando Inferencia Baseline (FinEval) ---")
    print(f"Modelo: {MODEL_ID}")
    print(f"Subsets de Compliance: {TARGET_SUBSETS}")
    
    # --- 2. Cargar Tokenizer y Modelo ---
    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo (torch.bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # --- 3. Iterar sobre Subsets y Generar ---
    print(f"Iniciando generación en {OUTPUT_FILE}...")
    
    # Abrimos el archivo en modo 'append' por si queremos correrlo por partes, 
    # pero lo limpiamos primero si es una ejecución nueva.
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass 

    total_processed = 0

    for subset in TARGET_SUBSETS:
        print(f"\n--- Procesando subset: {subset} ---")
        try:
            # Cargar el subset específico (ej. 'law')
            dataset = load_dataset(DATASET_ID, subset, split=DATASET_SPLIT, trust_remote_code=True)
        except Exception as e:
            print(f"Error cargando subset {subset}: {e}")
            continue

        print(f"Muestras en {subset}: {len(dataset)}")

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for sample in tqdm(dataset, desc=f"Inferencia {subset}"):
                
                # Preparar Prompt
                user_content = format_fineval_prompt(sample)
                ground_truth = sample['answer'] # Ej: "A", "B", etc.

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                
                try:
                    # Tokenizar
                    prompt_templated = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(prompt_templated, return_tensors="pt").to(model.device)

                    # Generar
                    outputs = model.generate(
                        **inputs,
                        do_sample=DO_SAMPLE,
                        temperature=INFERENCE_TEMP,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    # Decodificar
                    response_ids = outputs[0][inputs['input_ids'].shape[1]:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                    # Guardar resultado
                    result = {
                        "id": f"{subset}_{sample['id']}", # ID único combinando subset
                        "subset": subset,
                        "pregunta_completa": user_content,
                        "ground_truth": ground_truth,
                        "modelo_baseline": response_text
                    }
                    
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                    total_processed += 1

                except Exception as e:
                    print(f"Error en muestra {sample['id']}: {e}")
                    continue
    
    print(f"\n--- Completado. Total procesado: {total_processed} ---")
    print(f"Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()