import torch
import json
import re
import zipfile
import io
import requests
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuración ---
MODEL_ID = "SUFE-AIFLM-Lab/Fin-R1"
ZIP_URL = "https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEval.zip"
OUTPUT_FILE = "agent_outputs_langchain.jsonl"

# Parámetros
INFERENCE_TEMP = 0.6
MAX_NEW_TOKENS = 512

# --- 2. Definición de Herramientas (Mejoradas para "atrapar" más casos) ---

# Herramienta 1: Base de Datos Regulatoria (Añadido TAX y TARIFFS)
class RegulatoryInput(BaseModel):
    query: str = Field(description="Keywords about the regulation, tax, law or penalty (e.g., 'Company Law', 'VAT', 'Tariff Surcharge')")

@tool("lookup_regulatory_database", args_schema=RegulatoryInput)
def lookup_regulatory_database(query: str) -> str:
    """Consults the official financial regulations, TAX LAWS and PENALTY database."""
    q = query.lower()
    
    # Mock ampliado para cubrir tus ejemplos de Tax Law
    if "company law" in q or "shareholder" in q:
        return "[REGULATION_DB] Company Law Art. 103: Shareholders generally have pre-emptive rights to new shares."
    elif "basel" in q or "capital" in q:
        return "[REGULATION_DB] Basel III Accord: Minimum Common Equity Tier 1 (CET1) ratio is 4.5%."
    elif "insider" in q:
        return "[REGULATION_DB] Securities Law: Insider trading penalty: 5-10 years prison + fines up to 3x profit."
    elif "laundering" in q or "aml" in q:
        return "[REGULATION_DB] AML Act 2022: Cash transactions over $10k must be reported."
    
    # --- AÑADIDO PARA TUS CASOS DE EJEMPLO ---
    elif "tariff" in q or "surcharge" in q or "duty" in q:
        return "[REGULATION_DB] Customs Law Art 45: Tariff surcharge rate is 0.05% per day of delay. Formula: Amount * 0.0005 * Days."
    elif "city maintenance" in q or "construction tax" in q:
        return "[REGULATION_DB] Tax Law provisional reg: City maintenance and construction tax is based on the actual amount of VAT and Consumption Tax paid. Penalties are NOT included in the base."
    elif "consumption tax" in q or "vat" in q:
        return "[REGULATION_DB] Consumption Tax Law: Tax is levied on specific luxury items. It serves as a base for surcharges."
    # -----------------------------------------
    
    else:
        return "[REGULATION_DB] Standard fiduciary duties apply. Consult local statutes for specifics."

# Herramienta 2: Validador de Políticas (Sin cambios)
class PolicyInput(BaseModel):
    action_type: str = Field(description="The banking action")
    details: str = Field(description="Details")

@tool("check_internal_policy", args_schema=PolicyInput)
def check_internal_policy(action_type: str, details: str) -> str:
    """Checks if a proposed action violates internal bank policies."""
    content = f"{action_type} {details}".lower()
    if "loan" in content and "unsecured" in content:
        return "[POLICY_VALIDATOR] REJECTED. Policy 404: Unsecured loans > $50k require VP approval."
    if "gift" in content:
        return "[POLICY_VALIDATOR] WARNING. Policy 102: Gifts > $100 from clients must be declared."
    return "[POLICY_VALIDATOR] APPROVED. Action aligns with SOP."

# Herramienta 3: Calculadora (Sin cambios)
class CalcInput(BaseModel):
    expression: str = Field(description="Mathematical expression (e.g., '500 * 0.0005 * 26')")

@tool("financial_calculator", args_schema=CalcInput)
def financial_calculator(expression: str) -> str:
    """Evaluates mathematical expressions safely."""
    try:
        safe_expr = re.sub(r'[^0-9+\-*/().% ]', '', expression)
        return f"[CALCULATOR] Result: {eval(safe_expr)}"
    except:
        return "[CALCULATOR] Error."

# --- 3. Lógica del Agente ---

def parse_and_execute(llm_output: str) -> str:
    """Parsea la salida del LLM y ejecuta la herramienta"""
    # Regex más flexible por si el modelo pone espacios o saltos de línea
    match = re.search(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            tool_json = json.loads(match.group(1).strip())
            tool_name = tool_json.get("tool")
            tool_input = tool_json.get("input")
            
            if tool_name == "lookup_regulatory_database":
                res = lookup_regulatory_database.invoke(tool_input)
            elif tool_name == "check_internal_policy":
                res = check_internal_policy.invoke(tool_input)
            elif tool_name == "financial_calculator":
                res = financial_calculator.invoke(tool_input)
            else:
                res = "Error: Tool not found."
            
            return f"\nTOOL OBSERVATION: {res}\nBased on this evidence, the correct option is:"
        except:
            return "\nTOOL ERROR: JSON parsing failed.\n"
    return "" 

def main():
    print(f"--- Iniciando Inferencia AGENTE (MODO AGRESIVO) ---")
    
    # 1. Cargar Modelo
    print("⚙️ Cargando Fin-R1 (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # 2. Pipelines
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

    # 3. System Prompt "Tool-Forcing"
    # AQUÍ ESTÁ LA CLAVE: Instrucciones obligatorias y ejemplos (One-Shot)
    agent_system_prompt = """You are a strictly compliant AI Auditor. You MUST verify facts before answering.
    You have access to a database of Laws and a Calculator.

    RULES:
    1. If the question involves ANY law, tax, penalty, or regulation -> YOU MUST CALL lookup_regulatory_database.
    2. If the question involves calculation -> YOU MUST CALL financial_calculator.
    3. DO NOT answer from memory. Use the tools.

    FORMAT:
    To use a tool, output ONLY the JSON inside <tool_call> tags.

    EXAMPLE 1 (Law):
    User: What is the penalty for insider trading?
    Assistant: <tool_call> {"tool": "lookup_regulatory_database", "input": {"query": "insider trading penalty"}} </tool_call>

    EXAMPLE 2 (Math):
    User: Calculate 500 million * 0.05% * 20 days.
    Assistant: <tool_call> {"tool": "financial_calculator", "input": {"expression": "500 * 0.0005 * 20"}} </tool_call>

    Now, analyze the user's question and invoke the tool immediately if applicable.
    """

    # 4. Datos
    print("📥 Descargando FinEval.zip...")
    r = requests.get(ZIP_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    target_files = [f for f in z.namelist() if 'val' in f and f.endswith('.csv')]
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: pass
    
    total_processed = 0

    # 5. Bucle
    for filename in target_files:
        print(f"\n📄 Procesando: {filename}")
        with z.open(filename) as f_csv:
            df = pd.read_csv(f_csv)
        
        results_buffer = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Agent Thinking"):
            question = str(row.get('question', ''))
            options = ""
            for opt in ['A', 'B', 'C', 'D']:
                if opt in row and pd.notna(row[opt]): options += f"{opt}. {row[opt]}\n"
            
            # Forzamos al modelo a pensar que es una tarea de auditoría
            user_input = f"Question:\n{question}\n\nOptions:\n{options}\n\nAudit this question. Verify the law or calculate the values."

            messages = [
                {"role": "system", "content": agent_system_prompt},
                {"role": "user", "content": user_input}
            ]
            prompt_1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Paso 1: Generar Tool Call
            thought_output = llm_thinker.invoke(prompt_1)
            
            # Paso 2: Ejecutar Tool
            tool_feedback = parse_and_execute(thought_output)
            
            final_response = ""
            if tool_feedback:
                prompt_2 = prompt_1 + thought_output + tool_feedback
                final_response = llm_responder.invoke(prompt_2)
            else:
                final_response = thought_output.replace("<tool_call>", "").replace("</tool_call>", "")

            res = {
                "id": f"agent_{filename}_{idx}",
                "filename": filename,
                "pregunta": user_input,
                "ground_truth": str(row.get('answer', 'N/A')),
                "modelo_baseline": final_response, 
                "tool_used": bool(tool_feedback)
            }
            results_buffer.append(json.dumps(res, ensure_ascii=False))
            total_processed += 1
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for line in results_buffer: f_out.write(line + '\n')

    print(f"\n✅ Agente completado. Total: {total_processed}")

if __name__ == "__main__":
    main()