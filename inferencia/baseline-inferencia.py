import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 1. Constantes y Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"

# NUEVA FUENTE DE DATOS (Hugging Face Hub)
DATASET_ID = "TheFinAI/FINQA_test_test"
DATASET_SPLIT = "validation" # Basado en la vista 'val'

OUTPUT_FILE = "baseline_outputs_finqa_formatted.jsonl"

# Parámetros de Inferencia (Mantenemos la configuración del paper/experimento)
INFERENCE_TEMP = 0.6
DO_SAMPLE = True
MAX_NEW_TOKENS = 1024

# --- 2. System Prompt con Reglas de Formateo Estrictas ---
SYSTEM_PROMPT = """You are a helpful financial expert AI. 
You must solve the user's financial question using step-by-step chain-of-thought reasoning based strictly on the provided context.

CRITICAL OUTPUT FORMATTING RULES:
At the end of your response, you MUST provide the final answer wrapped in a LaTeX box: \\boxed{...}.
Analyze the type of question to determine the content of the box:

1. TYPE A: Numerical Questions (e.g., starts with 'What', 'Calculate', 'How much', 'What percentage').
   - The box must contain ONLY the numeric value (no words, no units inside).
   - Example: ... therefore the revenue grew by 500 million. \\boxed{500}
   - Example: ... the ratio is calculated as 0.465. \\boxed{0.465}

2. TYPE B: Boolean Questions (e.g., starts with 'Did', 'Was', 'Is').
   - The box must contain ONLY 'yes' or 'no' (lowercase).
   - Example: ... this indicates the company performed better. \\boxed{yes}

Do not use external tools. Use your internal reasoning capabilities."""

def format_finqa_prompt(row):
    """
    Construye el prompt combinando el contexto (pre_text + post_text) y la pregunta.
    FinQA tiene listas de strings en pre_text/post_text.
    """
    # Extraer y limpiar contexto
    pre_text = " ".join(row.get('pre_text', []))
    post_text = " ".join(row.get('post_text', []))
    
    # Construir el contexto completo
    full_context = f"{pre_text}\n{post_text}".strip()
    question = row.get('question', '')

    user_content = f"Context:\n{full_context}\n\nQuestion:\n{question}\n\nPlease analyze the context and provide the answer."
    return user_content

def main():
    print(f"--- Iniciando Inferencia BASELINE (FinQA) ---")
    print(f"Modelo: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID} (Split: {DATASET_SPLIT})")
    
    # --- 3. Cargar Tokenizer y Modelo ---
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

    # --- 4. Cargar Dataset desde Hugging Face ---
    print(f"Cargando dataset {DATASET_ID}...")
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
        print(f"Dataset cargado. Total de muestras: {len(dataset)}")
    except Exception as e:
        print(f"Error fatal cargando el dataset: {e}")
        return

    # Limpiar archivo de salida
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass 

    total_processed = 0

    # --- 5. Bucle de Inferencia ---
    # Iteramos directamente sobre el dataset cargado
    for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Inferencia Baseline"):
        
        # Preparar el contenido del usuario
        user_content = format_finqa_prompt(row)
        
        # Ground Truth (FinQA a veces tiene la respuesta en 'answer' como diccionario o string)
        ground_truth = row.get('answer', 'N/A')
        if isinstance(ground_truth, dict):
            # En FinQA original, 'answer' suele ser un dict con pasos de programa y resultado
            # Intentamos extraer el resultado final si existe esa estructura
            ground_truth = str(ground_truth)

        # Construir mensajes para el Chat Template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        try:
            # Tokenización
            prompt_templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_templated, return_tensors="pt").to(model.device)

            # Generación (Sin Herramientas - Pure CoT)
            outputs = model.generate(
                **inputs,
                do_sample=DO_SAMPLE,
                temperature=INFERENCE_TEMP,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decodificación
            response_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            # Estructura de salida compatible con el Juez
            result = {
                "id": row.get('id', f"finqa_val_{idx}"),
                "dataset_source": DATASET_ID,
                "pregunta": row.get('question', ''),
                "ground_truth": ground_truth,
                "modelo_baseline": response_text
            }
            
            # Guardar en tiempo real (append)
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            total_processed += 1

        except Exception as e:
            print(f"Error procesando muestra {idx}: {e}")
            continue

    print(f"\nInferencia Baseline Completada. Total procesado: {total_processed}")
    print(f"Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()