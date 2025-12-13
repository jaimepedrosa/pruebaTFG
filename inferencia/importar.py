import shutil
import os
import glob

# 1. Buscar el último archivo en /tmp
list_of_files = glob.glob('/tmp/batch_run_finqa_20251213_171619.jsonl')
if not list_of_files:
    print("❌ No se encontraron archivos en /tmp. Quizás se borraron.")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"📄 Archivo encontrado: {latest_file}")

# 2. CAMBIO CLAVE: Destino seguro en la raíz de tu usuario
# En lugar de la carpeta actual, usamos tu carpeta 'home'
user_home = os.path.expanduser("~") 
destination = "/home/jovyan/work/pruebaTFG/"

print(f"➡️ Intentando mover a: {destination}")

try:
    # Usamos copy en lugar de move para evitar errores de permisos de borrado en origen
    shutil.copy(latest_file, destination)
    print(f"✅ ¡ÉXITO! Archivo copiado a: {destination}")
    print("Busca este archivo en la carpeta principal de tu explorador de archivos.")
except Exception as e:
    print(f"❌ Falló de nuevo: {e}")
