# Análisis integral - adicción a redes sociales en jóvenes

## Contenido
- `analysis_pipeline.py`: ejecuta EDA, correlaciones, modelado y exporta resultados.
- `app.py`: dashboard interactivo en Dash.
- `outputs/`: tablas, métricas y reporte interactivo generado automáticamente.
- `adiccion_jovenes.xlsx`: base de datos original.

## Ejecución local
```bash
pip install -r requirements.txt
python analysis_pipeline.py
python app.py
```

## Uso en Binder
1. Abre el repositorio/proyecto en Binder.
2. Instala dependencias automáticamente con `requirements.txt`.
3. Ejecuta:
```bash
python analysis_pipeline.py
python app.py
```
4. Si usas un entorno Jupyter/Binder con proxy, expón el puerto 8050.

## Definición del objetivo binario
Para la regresión logística se define **alta adicción** como `Addicted_Score >= 7`.
Esta decisión se documenta para mantener interpretabilidad.

## Notas metodológicas
- `Student_ID` se excluye del modelado por ser un identificador.
- Se usan pipelines con imputación, escalamiento y codificación one-hot.
- El dashboard incluye filtros por edad, plataforma e impacto académico.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/https://mybinder.org/v2/gh/CristhianBravo/analisis-adiccion-jovenes/main?urlpath=proxy/8050/HEAD)