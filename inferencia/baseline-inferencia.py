import torch
import json
import datetime
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"
DATASET_ID = "TheFinAI/FINQA_test_test"
DATASET_SPLIT = "val"

# AJUSTA ESTO SEGÚN TU GPU:
BATCH_SIZE = 8 

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
    # El archivo se llamará batch_run_finqa_...jsonl
    filename = f"batch_run_finqa_{timestamp}.jsonl"
    
    # Intento 1: Carpeta actual (es '~/work' en tu caso)
    try:
        with open(filename, 'w', encoding='utf-8') as f: f.write("") 
        print(f"✅ Permisos correctos. Guardando en: {filename}")
        return filename
    except:
        # Intento 2: Carpeta temporal (/tmp)
        tmp_filename = os.path.join("/tmp", filename)
        print(f"⚠️ Sin permisos en la carpeta actual. Cambiando a: {tmp_filename}")
        return tmp_filename

def main():
    print(f"--- 🧪 INICIANDO PROCESAMIENTO POR LOTES (Batch: {BATCH_SIZE}) ---")
    
    # Esta función determinará si se guarda en ~/work o /tmp
    output_file = get_writable_output_file()

    # 1. Cargar Dataset
    print(f"📥 Cargando dataset...")
    try:
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=False)
        total_samples = len(dataset)
        print(f"✅ Total de muestras: {total_samples}")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return

    # 2. Cargar Modelo
    print("⚙️ Cargando modelo...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True,
            # attn_implementation="flash_attention_2" 
        )
        model.eval()
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # 3. Inferencia por Lotes
    print(f"🚀 Iniciando inferencia...")
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Procesando Lotes"):
            
            batch_indices = list(range(i, min(i + BATCH_SIZE, total_samples)))
            batch_rows = [dataset[idx] for idx in batch_indices]
            
            valid_rows = []
            prompts = []
            
            for row in batch_rows:
                query = row.get('query', '')
                if query:
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query}
                    ]
                    full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    prompts.append(full_prompt)
                    valid_rows.append(row)

            if not prompts: continue

            try:
                inputs = tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True 
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.6,
                        max_new_tokens=4096,
                        pad_token_id=tokenizer.pad_token_id
                    )

                generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
                responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                for row, response_text in zip(valid_rows, responses):
                    res = {
                        "id": row.get('id', 'unknown'),
                        "pregunta": row.get('query', ''),
                        "ground_truth": str(row.get('answer', 'N/A')),
                        "modelo_baseline": response_text.strip()
                    }
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                
                f_out.flush()

            except Exception as e:
                # Imprimimos el error, mostrando el índice del lote
                print(f"\n❌ Error en lote {i}-{i + BATCH_SIZE - 1}: {e}")

    print(f"\n✅ PROCESO COMPLETADO.")
    print(f"📄 Resultados: {output_file}")

if __name__ == "__main__":
    main()