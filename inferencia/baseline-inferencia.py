import torch
import json
import datetime
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 1. Constantes y Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"

# NUEVA FUENTE DE DATOS: FinQA desde Hugging Face
DATASET_ID = "TheFinAI/FINQA_test_test"
DATASET_SPLIT = "val"  # El viewer de HF muestra 'val'

# Parámetros de Inferencia (Igual que en el paper)
INFERENCE_TEMP = 0.6
DO_SAMPLE = True
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = "You are a helpful financial expert AI."

def get_writable_output_file():
    """
    Intenta encontrar una ruta donde tengamos permisos de escritura.
    Prueba primero localmente, luego en /tmp/.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_outputs_finqa_{timestamp}.jsonl"
    
    # Intento 1: Carpeta actual
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("") # Prueba de escritura
        print(f"✅ Permisos correctos. Guardando en: {filename}")
        return filename
    except PermissionError:
        print(f"⚠️ Sin permisos en la carpeta actual. Cambiando a /tmp/...")
    
    # Intento 2: Carpeta temporal (garantizado en Linux)
    tmp_filename = os.path.join("/tmp", filename)
    try:
        with open(tmp_filename, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"✅ Redirigido exitosamente a: {tmp_filename}")
        return tmp_filename
    except Exception as e:
        raise RuntimeError(f"❌ No se puede escribir ni en local ni en /tmp. Error: {e}")

def format_finqa_prompt(row):
    """
    Formatea la entrada específica de FinQA.
    """
    def clean_text(field):
        if isinstance(field, list):
            return " ".join([str(x) for x in field])
        return str(field) if field else ""

    pre_text = clean_text(row.get('pre_text', []))
    post_text = clean_text(row.get('post_text', []))
    question = row.get('question', '')
    
    full_context = f"{pre_text}\n{post_text}".strip()
    
    return f"Context:\n{full_context}\n\nQuestion:\n{question}\n\nPlease analyze the context and answer the question."

def main():
    print(f"--- Iniciando Inferencia Baseline (FinQA) ---")
    
    # --- 1. Determinar Archivo de Salida Seguro ---
    output_file = get_writable_output_file()
    
    # --- 2. Cargar Tokenizer y Modelo ---
    print("⚙️ Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("⚙️ Cargando modelo (torch.bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # --- 3. Cargar Dataset ---
    print(f"📥 Cargando dataset {DATASET_ID}...")
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
        print(f"✅ Dataset cargado. Muestras totales: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return

    # --- 4. Inferencia ---
    print("🚀 Ejecutando inferencia...")
    total_processed = 0
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        
        for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Procesando"):
            
            user_content = format_finqa_prompt(row)
            raw_answer = row.get('answer', 'N/A')
            ground_truth = str(raw_answer)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            try:
                prompt_templated = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt_templated, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    do_sample=DO_SAMPLE,
                    temperature=INFERENCE_TEMP,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                response_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                result = {
                    "id": row.get('id', f"finqa_{idx}"),
                    "dataset": DATASET_ID,
                    "pregunta": row.get('question', ''),
                    "ground_truth": ground_truth,
                    "modelo_baseline": response_text
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                total_processed += 1

            except Exception as e:
                print(f"⚠️ Error en muestra {idx}: {e}")
                continue

    print(f"\n✅ Completado. Procesados: {total_processed}")
    print(f"📄 RESULTADOS GUARDADOS EN: {output_file}")
    print("Copia esta ruta para usarla en el script del juez.")

if __name__ == "__main__":
    main()