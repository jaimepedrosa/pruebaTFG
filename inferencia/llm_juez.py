import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# --- 1. Configuración ---
# Lista de tuplas: (Nombre del Modelo, Archivo de Entrada)
FILES_TO_EVALUATE = [
    ("Baseline", "baseline_outputs_finqa_formatted.jsonl"), # Asegúrate de que los nombres coincidan con tu salida previa
    ("Agente", "agent_outputs_finqa_formatted.jsonl")
]

OUTPUT_FILE = "evaluation_comparison_strict_72B.jsonl" 

# Configuración del Modelo Juez
JUDGE_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct" 

# --- 2. Prompts del Paper (INTACTO - NO MODIFICAR) ---

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

# --- 3. Funciones Auxiliares de Validación y Extracción ---

def extract_boxed_content(text):
    """
    Función de Validación (Regex):
    Busca estrictamente el patrón LaTeX \boxed{contenido}.
    Retorna el contenido si existe, o None si falla el formato.
    """
    if not isinstance(text, str):
        return None
    
    # Busca \boxed{...} ignorando espacios entre boxed y la llave
    # Captura cualquier cosa que no sea una llave de cierre dentro
    pattern = r"\\boxed\s*\{([^}]+)\}"
    match = re.search(pattern, text)
    
    if match:
        return match.group(1).strip()
    return None

def extract_judgment(judge_output):
    """Extrae el 1 o 0 del formato boxed{{}} del Juez"""
    match = re.search(r"boxed\{(\d)\}", judge_output)
    if match:
        return int(match.group(1))
    
    if "1" in judge_output and "0" not in judge_output: return 1
    if "0" in judge_output and "1" not in judge_output: return 0
    return 0 

# --- 4. Main ---

def main():
    print(f"--- Iniciando Evaluación Estricta en DGX ---")
    print(f"Modelo Juez: {JUDGE_MODEL_ID}")
    
    # Contadores de Métricas
    stats = {
        "total_processed": 0,
        "sent_to_judge": 0,
        "discarded_total": 0,
        "discarded_details": {name: 0 for name, _ in FILES_TO_EVALUATE}
    }
    
    # Cargar Juez
    print("Cargando modelo Juez (Auto Device Map)...")
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
        print(f"Error cargando el modelo Juez: {e}")
        return

    # Limpiar archivo de salida
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: pass
    
    final_scores = {}

    # --- BUCLE DE EVALUACIÓN PARA CADA ARCHIVO ---
    for model_name, input_file in FILES_TO_EVALUATE:
        print(f"\nProcesando archivo: {model_name} ({input_file})")
        
        results = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line))
            print(f"   -> {len(results)} respuestas cargadas.")
        except FileNotFoundError:
            print(f"Archivo no encontrado. Saltando {model_name}...")
            final_scores[model_name] = "N/A"
            continue
        
        score_sum = 0
        total_evaluated_model = 0

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for item in tqdm(results, desc=f"Auditando {model_name}"):
                stats["total_processed"] += 1
                
                # Extraer respuesta del modelo a evaluar
                # Nota: Asumimos que ambos scripts guardaron la respuesta en 'modelo_baseline'
                # para mantener compatibilidad de claves, aunque sea el agente.
                model_response = item.get('modelo_baseline', '')
                
                # --- CORTAFUEGOS DE FORMATO (STRICT FORMAT FILTER) ---
                boxed_content = extract_boxed_content(model_response)
                
                if boxed_content is None:
                    # FALLO DE FORMATO: No enviamos al juez
                    stats["discarded_total"] += 1
                    stats["discarded_details"][model_name] += 1
                    
                    # Registramos el fallo en el log pero sin puntuación de juez
                    item['eval_status'] = "SKIPPED_FORMAT_ERROR"
                    item['score'] = 0 # Penalización por fallo de formato
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    continue
                
                # --- FORMATO VALIDO: ENVIAMOS AL JUEZ ---
                stats["sent_to_judge"] += 1
                
                # Preparamos el prompt usando SOLO el contenido de la caja para mayor precisión
                # o la respuesta completa si preferimos contexto (Paper usa respuesta completa).
                # Usaremos respuesta completa para seguir el paper fielmente.
                prompt_template = PROMPT_JUDGE_EN
                
                user_content = prompt_template.format(
                    ground_truth=item['ground_truth'],
                    model_response=model_response
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
                    
                    # Guardar resultados
                    item['eval_model_name'] = model_name
                    item['juez_raw'] = judge_response
                    item['score'] = score
                    item['eval_status'] = "JUDGED"
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f_out.flush()
                    
                    score_sum += score
                    total_evaluated_model += 1
                    
                except Exception as e:
                    print(f"Error invocando al juez: {e}")
                    continue
        
        # Calcular Accuracy Parcial (Sobre el total cargado, penalizando formatos inválidos)
        # Total Real = total_evaluated_model (enviados) + descartados (fallo formato)
        total_real = len(results) 
        accuracy = (score_sum / total_real) * 100 if total_real > 0 else 0
        final_scores[model_name] = accuracy
        print(f"Accuracy Parcial ({model_name}): {accuracy:.2f}% (Incluye penalización por formato)")

    # --- REPORTE DE MÉTRICAS Y DESCARTES ---
    print("\n" + "="*50)
    print("REPORTE DE FILTRADO Y EVALUACIÓN")
    print("="*50)
    print(f"Total Muestras Procesadas: {stats['total_processed']}")
    print(f"Enviadas al Juez (Formato OK): {stats['sent_to_judge']}")
    print(f"Descartadas (Formato Inválido): {stats['discarded_total']}")
    print("-" * 50)
    print("DESGLOSE DE FALLOS DE FORMATO:")
    for name, count in stats["discarded_details"].items():
        print(f"  - {name}: {count} muestras descartadas")
    print("="*50)
    
    print("\n" + "="*50)
    print("RESULTADOS FINALES (ACCURACY)")
    print("="*50)
    print(f"{'Modelo':<15} | {'Accuracy':<10}")
    print("-" * 28)
    for name, score in final_scores.items():
        if isinstance(score, (int, float)):
            print(f"{name:<15} | {score:.2f}%")
        else:
            print(f"{name:<15} | {score}")
    print("="*50)
    print(f"Detalles guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()