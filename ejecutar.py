# ejecutar.py

import sys
from pipeline import run_pipeline_from_excel_url

# 1. Definir la URL de tu archivo Excel
# **IMPORTANTE**: Reemplaza esta URL de ejemplo con la URL real de tu archivo Excel.
URL_DEL_EXCEL = "Archivos_xlsx/Planta_2023.xlsx"

# 2. Definir el random_state (opcional, por defecto es 42)
RANDOM_STATE = 42

try:
    print(f"Iniciando pipeline con URL: {URL_DEL_EXCEL} y random_state: {RANDOM_STATE}")
    
    # 3. Llamar a la función
    resultado = run_pipeline_from_excel_url(
        url=URL_DEL_EXCEL,
        random_state=RANDOM_STATE
    )
    
    # 4. Manejar o imprimir el resultado (ajusta esto según lo que devuelva tu función)
    print("\n--- Pipeline Finalizado con Éxito ---")
    # print(f"Resultado del pipeline: {resultado}") 
    
except Exception as e:
    print(f"\n--- Error al ejecutar el pipeline ---", file=sys.stderr)
    print(f"Detalle del error: {e}", file=sys.stderr)
    sys.exit(1)