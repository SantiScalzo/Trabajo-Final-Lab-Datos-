import pandas as pd

# Define el nombre de tu archivo.
nombre_archivo = 'Totalizadores_Planta_de_Cerveza_2021_2022-ConsolidadoKPI.csv'
print('hola')
# Carga el dataset en un 'DataFrame' de Pandas.
# Usa 'encoding' si el archivo tiene caracteres especiales (como acentos o 'ñ').
try:
    df = pd.read_csv(nombre_archivo, encoding='utf-8')
    
    # Muestra las primeras 5 filas para verificar que se cargó correctamente
    print("Dataset cargado exitosamente. Primeras filas:")
    print(df.head())
    
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")

# Ahora el dataset está en la variable 'df' y puedes trabajar con él.
