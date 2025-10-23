# PASO 1: IMPORTACIÓN DE LIBRERÍAS
# -------------------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# PASO 2 Y 3: CARGA DEL DATASET LOCAL
# -------------------------------------------------------------------------------------
# Se carga el archivo `data.json` desde la ruta local del proyecto.
# -------------------------------------------------------------------------------------
print("\nCargando dataset local...")
# Usando la ruta directa al archivo en tu proyecto
ruta_al_archivo = 'data.json'

dataset = load_dataset('json', data_files=ruta_al_archivo, split='train')
print("Dataset cargado exitosamente:", dataset)

# PASO 4: CONFIGURACIÓN DE QUANTIZACIÓN (BitsAndBytes)
# -------------------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# PASO 5: CARGA DEL MODELO Y TOKENIZER
# -------------------------------------------------------------------------------------
model_name = "SUFE-AIFLM-Lab/Fin-R1"

print(f"\nCargando el modelo base y el tokenizer: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("Modelo y tokenizer cargados exitosamente en 4-bits.")

# PASO 6: CONFIGURACIÓN DE LoRA (PEFT)
# -------------------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# PASO 7 Y 8: CONFIGURACIÓN E INICIALIZACIÓN DEL SFTTrainer
# -------------------------------------------------------------------------------------
print("\nConfigurando el SFTTrainer con SFTConfig...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
    args=SFTConfig(
        output_dir="./finr1-qlora-results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        optim="paged_adamw_8bit",
        fp16=True,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    ),
)
print("Trainer inicializado correctamente.")

# PASO 9: EJECUCIÓN DEL ENTRENAMIENTO
# -------------------------------------------------------------------------------------
print("\nIniciando el entrenamiento de QLoRA...")
trainer.train()
print("Entrenamiento completado.")

# PASO 10: GUARDAR EL ADAPTADOR LoRA
# -------------------------------------------------------------------------------------
adapter_path = "finr1-qlora-adapter"
trainer.save_model(adapter_path)
print(f"\n✅ Adaptador LoRA guardado exitosamente en: ./{adapter_path}")