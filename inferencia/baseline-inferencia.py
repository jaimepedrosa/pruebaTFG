import torch
import json
import datetime
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # Importamos para barra de progreso

# --- Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"
DATASET_ID = "TheFinAI/FINQA_test_test"
DATASET_SPLIT = "val"

# --- Configuración del Prompt ---
SYSTEM_PROMPT = (
    "You are a helpful financial expert AI. "
    "IMPORTANT: You must start your response IMMEDIATELY with the final result "
    "enclosed in LaTeX box format like \\boxed{answer}. "
    "After providing the boxed answer, explain your step-by-step reasoning."
)

def get_writable_output_file():
    """
    Intenta encontrar una ruta donde tengamos permisos de escritura.
    Prueba primero localmente, luego en /tmp/.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"full_run_finqa_{timestamp}.jsonl"
    
    tmp_filename = os.path.join("/tmp", filename)
    try:
        with open(tmp_filename, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"Redirigido exitosamente a: {tmp_filename}")
        return tmp_filename
    except Exception as e:
        raise RuntimeError(f"No se puede escribir ni en local ni en /tmp. Error: {e}")

def main():
    print(f"---  INICIANDO PROCESAMIENTO COMPLETO ---")
    
    # 1. Output File
    try:
        output_file = get_writable_output_file()
    except RuntimeError as e:
        print(e)
        return

    # 2. Cargar Dataset (Modo completo)
    print(f"Cargando dataset {DATASET_ID}...")
    try:
        # streaming=False descarga el dataset para poder contar el total de filas
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=False)
        total_samples = len(dataset)
        print(f"Dataset cargado. Total de muestras a procesar: {total_samples}")
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    # 3. Cargar Modelo
    print("Cargando modelo (bfloat16)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    # 4. Inferencia Completa
    print(f"Iniciando inferencia masiva...")
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        # Usamos tqdm para mostrar barra de progreso
        for i, row in tqdm(enumerate(dataset), total=total_samples, desc="Procesando"):
            
            query_content = row.get('query', '')
            ground_truth = str(row.get('answer', 'N/A'))
            
            if not query_content:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query_content}
            ]
            
            try:
                text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.6,
                    max_new_tokens=4096 
                )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                
                res = {
                    "id": row.get('id', f"sample_{i}"),
                    "pregunta": query_content,
                    "ground_truth": ground_truth,
                    "modelo_baseline": response
                }
                
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                f_out.flush() 
                
            except Exception as e:
                # Imprimimos el error pero NO detenemos el bucle para no perder todo el proceso
                print(f"\nError en muestra {i}: {e}")

    print(f"\nPROCESO COMPLETADO.")
    print(f"Resultados guardados en: {output_file}")

if __name__ == "__main__":
    main()