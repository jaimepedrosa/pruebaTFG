import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# --- 1. Configuración ---
# Lista de tuplas: (Nombre del Modelo, Archivo de Entrada)
FILES_TO_EVALUATE = [
    ("Baseline", "baseline_outputs_antfinance_zip.jsonl"),
    ("Agente", "agent_outputs_langchain.jsonl")
]

OUTPUT_FILE = "evaluation_comparison_72B.jsonl" 

# --- CAMBIO CRÍTICO PARA DGX ---
JUDGE_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct" 

# --- 2. Prompts del Paper (Sin cambios) ---

# Prompt en Inglés (Figura 10 - OF Format)
PROMPT_JUDGE_EN = """You are a scoring assistant for financial questions. I will provide you with a
<ground truth> and a <model answer>. Please determine whether the <model answer> has the
same meaning as the <ground truth> according to the following rules. If they are
consistent, output 1, otherwise output 0.

<ground truth>
{ground_truth}
<ground truth>

<model answer>
{model_response}
<model answer>

### Rules:
1. If the <ground truth> is a numerical value, and the format of the <model answer> is
different from that of the <ground truth>, but the numerical values are the same, then
it is considered that the meanings are consistent.
2. If the <ground truth> is a numerical value, and the final result of the <model answer> is
consistent with the <ground truth> after rounding, then it is considered that the meanings
are consistent.

### Output Format:
Make the judgment according to the above rules, and finally put the judgment result 1 or 0
in boxed{{}}, for example, boxed{{1}} or boxed{{0}}"""

# Prompt en Chino (Figura 14 - FinEval es mayormente chino)
PROMPT_JUDGE_ZH = """你是一个金融题目结果评分助手,我会给你一个<标准答案>与一个<模型回答>,请根据以下规则判断<模型回答>是否与
<标准答案>的含义一致。如果一致,输出1,否则输出0。

<标准答案>
{ground_truth}
<标准答案>

<模型回答>
{model_response}
<模型回答>

### 规则:
1. 如果<标准答案>是一个数值,<模型回答>与<标准答案>的格式不一样,但是数值一致,则认为含义一致。
2. 如果<标准答案>是一个数值,<模型回答>的最终结果经四舍五入后与<标准答案>一致,则认为含义一致。

### 回复格式:
按照以上规则给出判断,并在最后将判断结果1 or 0放在boxed{{}}中,例如boxed{{1}}或boxed{{0}}"""

def extract_judgment(judge_output):
    """Extrae el 1 o 0 del formato boxed{{}}"""
    match = re.search(r"boxed\{(\d)\}", judge_output)
    if match:
        return int(match.group(1))
    
    if "1" in judge_output and "0" not in judge_output: return 1
    if "0" in judge_output and "1" not in judge_output: return 0
    return 0 

def main():
    print(f"--- Iniciando Evaluación Comparativa en DGX ---")
    print(f"Modelo Juez: {JUDGE_MODEL_ID} (72B Parámetros)")
    
    # Cargar Juez
    print("⚖️ Cargando modelo Juez (Auto Device Map)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True
        )
        print(f"Distribución del modelo: {model.hf_device_map}")
    except Exception as e:
        print(f"❌ Error cargando el modelo Juez: {e}")
        return

    # Limpiar archivo de salida
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: pass
    
    final_scores = {}

    # --- BUCLE DE EVALUACIÓN PARA CADA ARCHIVO ---
    for model_name, input_file in FILES_TO_EVALUATE:
        print(f"\n📂 Procesando archivo: {model_name} ({input_file})")
        
        results = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line))
            print(f"   -> {len(results)} respuestas cargadas.")
        except FileNotFoundError:
            print(f"   ⚠️ Archivo no encontrado. Saltando {model_name}...")
            final_scores[model_name] = "N/A"
            continue
        
        score_sum = 0
        total_evaluated = 0

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for item in tqdm(results, desc=f"Juzgando {model_name}"):
                # Detectar idioma (Chino vs Inglés)
                has_chinese = bool(re.search(r'[\u4e00-\u9fff]', str(item.get('ground_truth', ''))))
                prompt_template = PROMPT_JUDGE_ZH if has_chinese else PROMPT_JUDGE_EN
                
                # Nota: En ambos archivos guardamos la respuesta en la clave 'modelo_baseline'
                # para mantener compatibilidad, aunque en el archivo del agente sea la respuesta del agente.
                user_content = prompt_template.format(
                    ground_truth=item['ground_truth'],
                    model_response=item['modelo_baseline']
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content}
                ]
                
                try:
                    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1, # Determinista
                        do_sample=False
                    )
                    
                    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    judge_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    score = extract_judgment(judge_response)
                    
                    # Enriquecer el objeto para el log final
                    item['eval_model_name'] = model_name
                    item['juez_raw'] = judge_response
                    item['score'] = score
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f_out.flush()
                    
                    score_sum += score
                    total_evaluated += 1
                    
                except Exception as e:
                    print(f"Error evaluando fila: {e}")
                    continue
        
        # Calcular Accuracy Parcial
        accuracy = (score_sum / total_evaluated) * 100 if total_evaluated > 0 else 0
        final_scores[model_name] = accuracy
        print(f"📊 Accuracy Parcial ({model_name}): {accuracy:.2f}%")

    # --- REPORTE FINAL ---
    print("\n" + "="*40)
    print("🏆 RESULTADOS FINALES DE LA COMPARATIVA")
    print("="*40)
    print(f"{'Modelo':<15} | {'Accuracy':<10}")
    print("-" * 28)
    for name, score in final_scores.items():
        if isinstance(score, (int, float)):
            print(f"{name:<15} | {score:.2f}%")
        else:
            print(f"{name:<15} | {score}")
    print("="*40)
    print(f"Detalles guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


"""
📂 Procesando archivo: Baseline (baseline_outputs_antfinance_zip.jsonl)
   -> 1151 respuestas cargadas.
Juzgando Baseline:   0%|                                                              | 0/1151 [00:00<?, ?it/s]The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Juzgando Baseline: 100%|███████████████████████████████████████████████████| 1151/1151 [09:20<00:00,  2.05it/s]
📊 Accuracy Parcial (Baseline): 67.68%

📂 Procesando archivo: Agente (agent_outputs_langchain.jsonl)
   -> 1151 respuestas cargadas.
Juzgando Agente: 100%|█████████████████████████████████████████████████████| 1151/1151 [08:35<00:00,  2.23it/s]
📊 Accuracy Parcial (Agente): 44.22%

========================================
🏆 RESULTADOS FINALES DE LA COMPARATIVA
========================================
Modelo          | Accuracy  
----------------------------
Baseline        | 67.68%
Agente          | 44.22%
========================================
Detalles guardados en evaluation_comparison_72B.jsonl
"""    