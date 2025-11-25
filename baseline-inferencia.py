import torch
import json
import pandas as pd
import requests
import zipfile
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 1. Constantes y Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"

# URL Directa al ZIP del dataset
ZIP_URL = "https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEval.zip"

OUTPUT_FILE = "baseline_outputs_antfinance_zip.jsonl"

# Parámetros del Paper
INFERENCE_TEMP = 0.6
DO_SAMPLE = True
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = "You are a helpful financial expert AI."

def format_prompt_generic(row):
    """
    Formatea la entrada de manera genérica intentando detectar las columnas.
    """
    # Detectar pregunta
    question = str(row.get('question', row.get('input', '')))
    
    # Detectar opciones
    options = ""
    for opt_label in ['A', 'B', 'C', 'D', 'E']:
        if opt_label in row and pd.notna(row[opt_label]):
            options += f"{opt_label}. {row[opt_label]}\n"
    
    return f"Question:\n{question}\n\nOptions:\n{options}\n\nPlease analyze the question and select the correct option."

def main():
    print(f"--- Iniciando Inferencia Baseline (Modo: ZIP DIRECTO) ---")
    
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

    # --- 3. Descargar y Procesar ZIP ---
    print(f"📥 Descargando FinEval.zip desde HuggingFace...")
    try:
        response = requests.get(ZIP_URL)
        response.raise_for_status()
        print("✅ Descarga completada.")
    except Exception as e:
        print(f"❌ Error descargando el ZIP: {e}")
        return

    # Abrir el ZIP en memoria
    print("📂 Abriendo archivo ZIP...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Listar archivos dentro del ZIP
        all_files = z.namelist()
        
        # Filtrar solo los CSV de validación ('val')
        # Buscamos archivos que tengan 'val' en el nombre y terminen en .csv
        target_files = [f for f in all_files if 'val' in f and f.endswith('.csv')]
        
        print(f"✅ Se encontraron {len(target_files)} archivos CSV dentro del ZIP.")
        
        if len(target_files) == 0:
            print("⚠️ No se encontraron archivos 'val' en el ZIP. Lista de archivos disponibles:", all_files[:5])
            return

        # Limpiar archivo de salida
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass 
        
        total_processed = 0

        # --- 4. Iterar sobre cada CSV dentro del ZIP ---
        for filename in target_files:
            print(f"\n📄 Procesando: {filename}")
            
            try:
                # Leer CSV desde el ZIP
                with z.open(filename) as csv_file:
                    df = pd.read_csv(csv_file)
                
                print(f"   -> Filas encontradas: {len(df)}")
                
                results_buffer = []
                
                # Iterar filas del CSV
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inferencia"):
                    user_content = format_prompt_generic(row)
                    ground_truth = str(row.get('answer', 'N/A'))

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
                            "id": f"{filename}_{idx}",
                            "filename": filename,
                            "pregunta_completa": user_content,
                            "ground_truth": ground_truth,
                            "modelo_baseline": response_text
                        }
                        
                        results_buffer.append(json.dumps(result, ensure_ascii=False))
                        total_processed += 1

                    except Exception as e:
                        continue # Si falla una fila, seguimos
                
                # Guardar al archivo
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                    for line in results_buffer:
                        f_out.write(line + '\n')
                        
            except Exception as e:
                print(f"⚠️ Error leyendo archivo {filename}: {e}")
                continue

    print(f"\n✅ Completado. Total procesado: {total_processed}")
    print(f"Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()