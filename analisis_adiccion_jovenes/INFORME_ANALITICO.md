
# Informe analítico integral: adicción a redes sociales en jóvenes

## 1. Resumen ejecutivo

La base contiene **705 registros** y **13 variables originales**.  
No se detectaron **valores nulos** ni **duplicados completos**. Esto simplifica el modelado y permite interpretar los resultados sin sesgos derivados de imputación o desduplicación.

Los hallazgos más fuertes fueron:

- **Mental_Health_Score** presenta una correlación de **-0.945** con `Addicted_Score` (inversa y muy fuerte).
- **Conflicts_Over_Social_Media** presenta una correlación de **0.934**.
- **Avg_Daily_Usage_Hours** presenta una correlación de **0.832**.
- **Sleep_Hours_Per_Night** presenta una correlación de **-0.765**.

En modelado predictivo:

- La **regresión lineal** obtuvo **R²=0.958**, **RMSE=0.323** y **MAE=0.221**.
- La **regresión logística** para clasificar **alta adicción** (`Addicted_Score >= 7`) obtuvo **accuracy=0.993**, **F1=0.994** y **ROC-AUC=0.998**.

## 2. Calidad de datos

- Total de nulos: **0**
- Duplicados completos: **0**
- La variable `Student_ID` fue tratada como identificador y excluida del modelado.

### Outliers detectados por IQR
| variable                    |   lower |   upper |   count |        pct |
|:----------------------------|--------:|--------:|--------:|-----------:|
| Age                         |   14.5  |   26.5  |       0 | 0          |
| Avg_Daily_Usage_Hours       |    1.55 |    8.35 |       3 | 0.00425532 |
| Sleep_Hours_Per_Night       |    3.45 |   10.25 |       0 | 0          |
| Mental_Health_Score         |    2    |   10    |       0 | 0          |
| Conflicts_Over_Social_Media |   -1    |    7    |       0 | 0          |
| Addicted_Score              |    0.5  |   12.5  |       0 | 0          |

Interpretación:
- Solo `Avg_Daily_Usage_Hours` mostró valores atípicos por IQR y en una proporción muy baja.
- No se observan anomalías masivas que obliguen a winsorización o transformación inmediata.

## 3. Estadística descriptiva

Variables numéricas relevantes:

| variable                    |     mean |   50% |      std |   min |   max |
|:----------------------------|---------:|------:|---------:|------:|------:|
| Age                         | 20.6596  |  21   | 1.39922  |  18   |  24   |
| Avg_Daily_Usage_Hours       |  4.91872 |   4.8 | 1.25739  |   1.5 |   8.5 |
| Sleep_Hours_Per_Night       |  6.86894 |   6.9 | 1.12685  |   3.8 |   9.6 |
| Mental_Health_Score         |  6.22695 |   6   | 1.10506  |   4   |   9   |
| Conflicts_Over_Social_Media |  2.84965 |   3   | 0.957968 |   0   |   5   |
| Addicted_Score              |  6.43688 |   7   | 1.58716  |   2   |   9   |

Interpretación:
- La edad está concentrada en un rango estrecho, lo que limita su poder explicativo.
- El uso diario promedio ronda las **4.92 horas**.
- El puntaje medio de adicción es **6.44**, con mayor concentración entre 7 y 8.

## 4. Relaciones y correlaciones

Las relaciones más relevantes respecto a `Addicted_Score` fueron:

- `Mental_Health_Score`: **-0.945**
- `Conflicts_Over_Social_Media`: **0.934**
- `Avg_Daily_Usage_Hours`: **0.832**
- `Sleep_Hours_Per_Night`: **-0.765**
- `Age`: **-0.166**

Interpretación:
- A mayor uso diario y mayor nivel de conflictos por redes, aumenta el score de adicción.
- A mejor salud mental y mayor cantidad de sueño, disminuye el score de adicción.
- La edad tiene una relación débil y probablemente secundaria en esta muestra.

## 5. Modelado estadístico y predictivo

### 5.1 Regresión lineal

Métricas:
- **R²:** 0.958
- **RMSE:** 0.323
- **MAE:** 0.221
- **CV R² promedio:** 0.967

Variables más influyentes por importancia de permutación:
| variable                     |   importancia |
|:-----------------------------|--------------:|
| Mental_Health_Score          |   0.283317    |
| Conflicts_Over_Social_Media  |   0.109101    |
| Affects_Academic_Performance |   0.0753673   |
| Country                      |   0.0399213   |
| Avg_Daily_Usage_Hours        |   0.00993999  |
| Most_Used_Platform           |   0.00504689  |
| Relationship_Status          |   0.00352833  |
| Gender                       |   0.000394893 |

Coeficientes más intensos:
| variable                         |   coeficiente |
|:---------------------------------|--------------:|
| cat__Country_Kuwait              |     -1.4263   |
| cat__Country_Bhutan              |     -1.21914  |
| cat__Country_Uruguay             |      0.979854 |
| cat__Country_Cyprus              |      0.962556 |
| cat__Country_Egypt               |      0.953546 |
| cat__Country_Serbia              |      0.917783 |
| cat__Country_Slovakia            |     -0.849514 |
| cat__Most_Used_Platform_Snapchat |      0.848077 |
| cat__Country_South Korea         |     -0.768936 |
| cat__Country_Ghana               |     -0.751988 |

Interpretación:
- La importancia de permutación sugiere que el mayor peso práctico está en `Mental_Health_Score` y `Conflicts_Over_Social_Media`.
- Los coeficientes de categorías específicas deben interpretarse con cautela porque muchas categorías, especialmente países, pueden estar representadas por pocos casos.

### 5.2 Regresión logística

Objetivo binario:
- **Alta adicción = 1** cuando `Addicted_Score >= 7`

Métricas:
- **Accuracy:** 0.993
- **Precision:** 0.988
- **Recall:** 1.000
- **F1:** 0.994
- **ROC-AUC:** 0.998

Variables más influyentes por importancia de permutación:
| variable                     |   importancia |
|:-----------------------------|--------------:|
| Mental_Health_Score          |   0.155023    |
| Country                      |   0.00142621  |
| Sleep_Hours_Per_Night        |   0.00113683  |
| Most_Used_Platform           |   0.000764779 |
| Affects_Academic_Performance |   0.000744109 |
| Relationship_Status          |   0.0006821   |
| Conflicts_Over_Social_Media  |   0.000434064 |
| Age                          |   0.000165358 |

Coeficientes logísticos más intensos:
| variable                              |   coeficiente |   odds_ratio |
|:--------------------------------------|--------------:|-------------:|
| num__Mental_Health_Score              |     -3.76032  |    0.0232762 |
| cat__Country_South Korea              |     -1.62223  |    0.197458  |
| cat__Most_Used_Platform_KakaoTalk     |     -1.62185  |    0.197534  |
| num__Sleep_Hours_Per_Night            |     -1.20199  |    0.300597  |
| cat__Country_Spain                    |      1.10435  |    3.01727   |
| cat__Affects_Academic_Performance_Yes |      1.00023  |    2.7189    |
| cat__Affects_Academic_Performance_No  |     -0.973076 |    0.377919  |
| num__Conflicts_Over_Social_Media      |      0.86525  |    2.3756    |
| num__Avg_Daily_Usage_Hours            |      0.744377 |    2.10513   |
| cat__Country_Ireland                  |     -0.681427 |    0.505894  |

Interpretación:
- El modelo separa casi perfectamente los casos de alta adicción.
- `Mental_Health_Score` vuelve a emerger como el principal predictor.
- Dado el rendimiento casi perfecto, es recomendable validar si la estructura del dataset ya incorpora reglas fuertes de construcción.

## 6. Hallazgos clave

1. **El eje principal del fenómeno es psicosocial**, no demográfico.  
   Edad y género aportan mucho menos que salud mental, horas de uso y conflictos sociales.

2. **El sueño funciona como variable protectora**.  
   Menores horas de sueño se asocian con mayor adicción.

3. **El impacto académico aparece alineado con adicción elevada**.  
   Esto refuerza la hipótesis de efecto funcional real sobre desempeño.

4. **El rendimiento predictivo es demasiado alto para un escenario social típico**.  
   Esto puede ser señal de:
   - datos sintéticos,
   - reglas de generación muy determinísticas,
   - o una muestra donde las variables explicativas están muy cerca del constructo objetivo.

## 7. Recomendaciones accionables

- Priorizar intervenciones sobre estudiantes con:
  - bajo `Mental_Health_Score`,
  - alta frecuencia de conflictos por redes,
  - más horas de uso diario,
  - menos horas de sueño.

- Diseñar un sistema de monitoreo temprano con estas variables como **señales de riesgo**.

- Validar con expertos de dominio si `Addicted_Score` fue calculado a partir de variables muy cercanas, porque eso puede inflar la capacidad predictiva.

- Si el objetivo es despliegue real, realizar:
  - validación externa,
  - prueba temporal,
  - y auditoría de sesgo por país/plataforma.

## 8. Entregables incluidos

- `analysis_pipeline.py`
- `app.py`
- `outputs/reporte_interactivo.html`
- tablas CSV con descriptivos, correlaciones, outliers y coeficientes
- dashboard Dash listo para ejecutar
