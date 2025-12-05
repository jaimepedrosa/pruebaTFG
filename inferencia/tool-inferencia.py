import torch
import json
import re
import os
import datetime
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from datasets import load_dataset

# --- 1. Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"

# Fuente de datos (Hugging Face Hub)
DATASET_ID = "TheFinAI/FINQA_test_test"
DATASET_SPLIT = "val" 

# Parámetros de Inferencia
INFERENCE_TEMP = 0.6
MAX_NEW_TOKENS = 512

def get_writable_output_file():
    """
    Gestiona permisos de escritura de forma robusta.
    Prueba local -> Falla -> Prueba /tmp/
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_outputs_finqa_{timestamp}.jsonl"
    
    # Intento 1: Carpeta local
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("") # Test de escritura
        print(f"✅ Permisos correctos. Guardando en: {filename}")
        return filename
    except PermissionError:
        print(f"⚠️ Sin permisos locales. Cambiando a /tmp/...")
        
    # Intento 2: Carpeta temporal
    tmp_filename = os.path.join("/tmp", filename)
    try:
        with open(tmp_filename, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"✅ Redirigido exitosamente a: {tmp_filename}")
        return tmp_filename
    except Exception as e:
        raise RuntimeError(f"❌ Error crítico de escritura: {e}")

# --- 2. Definición de Esquemas de Entrada (Pydantic) ---

class PercentageChangeInput(BaseModel):
    new_value: float = Field(description="The value from the more recent period")
    old_value: float = Field(description="The value from the older/base period")

class FinancialRatioInput(BaseModel):
    numerator: float = Field(description="The value to be divided (the part)")
    denominator: float = Field(description="The value to divide by (the whole or base)")

class AggregationInput(BaseModel):
    values: List[float] = Field(description="A list of numbers to operate on")
    operation: str = Field(description="The operation to perform: 'sum', 'difference', 'average'")

# --- 3. Implementación de Herramientas (FinQA Tool Suite) ---

@tool("calculate_percentage_change", args_schema=PercentageChangeInput)
def calculate_percentage_change(new_value: float, old_value: float) -> str:
    """
    Use this tool when the user asks for the growth rate, percentage increase, 
    percentage decrease, or percentage change between two values over time 
    (e.g., 'growth from 2012 to 2013'). It calculates ((new_value - old_value) / old_value).
    """
    try:
        if old_value == 0:
            return "[MATH_ERROR] Cannot calculate percentage change when old_value is 0."
        
        result = (new_value - old_value) / old_value
        result_percent = result * 100
        return f"[CALCULATION] (({new_value} - {old_value}) / {old_value}) = {result:.4f} ({result_percent:.2f}%)"
    except Exception as e:
        return f"[MATH_ERROR] Error calculating percentage change: {str(e)}"

@tool("calculate_financial_ratio", args_schema=FinancialRatioInput)
def calculate_financial_ratio(numerator: float, denominator: float) -> str:
    """
    Use this tool to calculate ratios, margins, or the percentage that one value 
    represents of another (e.g., 'what percent of total revenue is net income', 
    'operating margin', or 'ratio of debt to equity'). Can also be used to find 
    a total given a part and its percentage.
    """
    try:
        if denominator == 0:
            return "[MATH_ERROR] Division by zero. Denominator is 0."
        
        result = numerator / denominator
        result_percent = result * 100
        return f"[CALCULATION] ({numerator} / {denominator}) = {result:.4f} ({result_percent:.2f}%)"
    except Exception as e:
        return f"[MATH_ERROR] Error calculating ratio: {str(e)}"

@tool("perform_basic_aggregation", args_schema=AggregationInput)
def perform_basic_aggregation(values: List[float], operation: str) -> str:
    """
    Use this tool to calculate the sum, difference, or average of a list of numbers. 
    Useful for questions asking for 'total amount over years', 'combined value', 
    'difference between', or 'net change' in absolute terms (not percentage).
    """
    try:
        if not values:
            return "[MATH_ERROR] No values provided for aggregation."
        
        op = operation.lower().strip()
        
        if op == 'sum':
            res = sum(values)
            return f"[CALCULATION] Sum of {values} = {res}"
        
        elif op == 'average':
            res = sum(values) / len(values)
            return f"[CALCULATION] Average of {values} = {res}"
        
        elif op == 'difference':
            # Asume resta secuencial: v[0] - v[1] - v[2]...
            res = values[0]
            for v in values[1:]:
                res -= v
            return f"[CALCULATION] Difference of {values} = {res}"
        
        else:
            return f"[MATH_ERROR] Unsupported operation '{op}'. Use 'sum', 'average', or 'difference'."
            
    except Exception as e:
        return f"[MATH_ERROR] Error performing aggregation: {str(e)}"

# --- 4. Lógica del Agente (Router y Ejecución) ---

def parse_and_execute(llm_output: str) -> str:
    """Parsea la salida del LLM, busca <tool_call> y ejecuta la herramienta correspondiente."""
    # Regex robusta para capturar JSON multilinea
    match = re.search(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        try:
            tool_content = match.group(1).strip()
            tool_json = json.loads(tool_content)
            
            tool_name = tool_json.get("tool")
            tool_input = tool_json.get("input")
            
            # Router de herramientas financieras
            if tool_name == "calculate_percentage_change":
                res = calculate_percentage_change.invoke(tool_input)
            elif tool_name == "calculate_financial_ratio":
                res = calculate_financial_ratio.invoke(tool_input)
            elif tool_name == "perform_basic_aggregation":
                res = perform_basic_aggregation.invoke(tool_input)
            else:
                res = f"Error: Tool '{tool_name}' not found in registry."
            
            return f"\nTOOL OBSERVATION: {res}\nBased on this calculation, determine the final answer and format it with \\boxed{{}}."
        except json.JSONDecodeError:
            return "\nTOOL ERROR: JSON parsing failed. Ensure valid JSON format inside tags.\n"
        except Exception as e:
            return f"\nTOOL ERROR: Execution failed: {str(e)}\n"
            
    return "" # No hubo llamada a herramienta

def main():
    print(f"--- Iniciando Inferencia AGENTE FINANCIERO (Format Mode: Boxed) ---")
    
    # 1. Determinar Salida Segura
    output_file = get_writable_output_file()
    
    # 2. Cargar Modelo
    print("⚙️ Cargando Fin-R1 (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # 3. Pipelines
    thought_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=256, do_sample=True, temperature=INFERENCE_TEMP,
        pad_token_id=tokenizer.pad_token_id, return_full_text=False
    )
    llm_thinker = HuggingFacePipeline(pipeline=thought_pipe)

    response_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=512, do_sample=True, temperature=INFERENCE_TEMP,
        pad_token_id=tokenizer.pad_token_id, return_full_text=False
    )
    llm_responder = HuggingFacePipeline(pipeline=response_pipe)

    # 4. System Prompt "Math-Reasoning + Strict Formatting"
    agent_system_prompt = """You are a financial expert AI specializing in numerical reasoning. 
    You have access to a suite of precise calculation tools. Use them to ensure accuracy.
    
    TOOLS AVAILABLE:
    1. calculate_percentage_change(new_value, old_value): For growth rates, % increase/decline.
    2. calculate_financial_ratio(numerator, denominator): For margins, ratios, proportions.
    3. perform_basic_aggregation(values, operation): For sums, differences, or averages of lists.

    CRITICAL OUTPUT FORMATTING RULES:
    After performing your analysis and utilizing tools, you MUST conclude with a final answer wrapped in a LaTeX box: \\boxed{...}.
    Analyze the type of question to determine the content of the box:

    TYPE A: Numerical Questions (e.g., 'What is...', 'Calculate...', 'How much...', 'What percentage...')
    - The box must contain ONLY the numeric value (no words inside).
    - Example User: 'What percentage of total net revenue in 2015 was net interest income?'
    - Example Output: The calculated value is 0.46513. \\boxed{0.46513}

    TYPE B: Boolean Questions (e.g., 'Did...', 'Was...', 'Is...', 'Do...')
    - The box must contain ONLY 'yes' or 'no' (lowercase).
    - Example User: 'Did the series c outperform the s&p 500?'
    - Example Output: Based on the data, the series c performed better. \\boxed{yes}

    PROCESS:
    1. Analyze the context and question.
    2. Use tools if calculation is needed (output <tool_call> JSON).
    3. State your reasoning.
    4. Provide the final answer ending with \\boxed{...}.
    """

    # 5. Datos (Carga desde Hugging Face Hub)
    print(f"📥 Cargando Dataset desde Hugging Face: {DATASET_ID} (Split: {DATASET_SPLIT})...")
    try:
        # Carga del dataset usando la librería datasets
        dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
        print(f"✅ Dataset cargado. Total de muestras: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error cargando el dataset desde Hugging Face: {e}")
        return

    total_processed = 0

    # 6. Bucle de Inferencia
    with open(output_file, 'a', encoding='utf-8') as f_out:
        
        for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Agent Reasoning"):
            
            question = row.get('question', '')
            
            # Construimos el contexto
            context_text = row.get('pre_text', [])
            if isinstance(context_text, list):
                context_text = " ".join(context_text)
            
            user_input = f"Context:\n{context_text}\n\nQuestion:\n{question}\n\nAnalyze the data, calculate if necessary, and provide the final answer in a \\boxed{{}}."

            messages = [
                {"role": "system", "content": agent_system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            try:
                prompt_1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # --- Paso 1: Pensar / Llamar Herramienta ---
                thought_output = llm_thinker.invoke(prompt_1)
                
                # --- Paso 2: Ejecutar Herramienta (Si aplica) ---
                tool_feedback = parse_and_execute(thought_output)
                
                final_response = ""
                if tool_feedback:
                    # Inyectar resultado
                    prompt_2 = prompt_1 + thought_output + tool_feedback
                    # Invocamos de nuevo
                    final_response = llm_responder.invoke(prompt_2)
                else:
                    final_response = thought_output.replace("<tool_call>", "").replace("</tool_call>", "")

                # Construimos el resultado
                sample_id = row.get('id', f"finqa_val_{idx}")
                ground_truth = row.get('answer', 'N/A')
                if isinstance(ground_truth, dict):
                    ground_truth = str(ground_truth)

                res = {
                    "id": sample_id,
                    "dataset_source": DATASET_ID,
                    "pregunta": question,
                    "ground_truth": ground_truth,
                    "modelo_baseline": final_response, # Clave compatible con el Juez
                    "tool_used": bool(tool_feedback)
                }
                
                # Escribir en modo append (línea a línea)
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                f_out.flush() # Guardar progreso
                
                total_processed += 1
                
            except Exception as e:
                print(f"⚠️ Error procesando muestra {idx}: {e}")
                continue

    print(f"\n✅ Inferencia completada. Total procesado: {total_processed}")
    print(f"📄 Resultados guardados en: {output_file}")
    print("Recuerda usar el script de recuperación si se guardó en /tmp")

if __name__ == "__main__":
    main()