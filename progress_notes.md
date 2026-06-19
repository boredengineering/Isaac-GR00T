# PRUEBA DE CONCEPTO
## Cosmos 3 + GR00T N + Unitree G1
**Tarea:** Pick · Carry · Place — Integración ROS2 / NAV2 / Isaac Sim  
**Fecha:** Junio 2026 — Confidencial  

## Tabla de contenidos
*(Tabla de contenidos omitida para brevedad)*

## 1. Objetivo y alcance de la prueba
Este documento define el plan técnico completo para la primera prueba de Cosmos 3 integrado con Isaac GR00T N1.6 sobre el robot humanoide Unitree G1.
El objetivo es validar el pipeline end-to-end de generación de datos sintéticos (SDG) + fine-tuning + deployment para la tarea autónoma de pick-carry-place de una botella de agua.

La prueba opera en dos entornos en paralelo:
* **Entorno físico:** Unitree G1 real con el stack de navegación autónoma NAV2 + ROS2 Humble ya integrado y en fase de pruebas
* **Entorno de simulación:** Isaac Sim con el Digital Twin del espacio de prueba y del propio G1

El sistema de navegación NAV2 + ROS2 se encuentra integrado al G1 y en fase de pruebas, por lo que esta prueba se enfoca en añadir las capacidades de manipulación inteligente mediante GR00T N1.6, sin reemplazar el stack de locomoción existente.

### 1.1 Criterio de éxito

| Criterio | Objetivo mínimo | Objetivo deseable |
|---|---|---|
| Task success rate (end-to-end) | ≥ 50% en 20 intentos | ≥ 70% |
| Grasp success rate | ≥ 70% | ≥ 85% |
| Drop rate durante carry | ≤ 30% | ≤ 10% |
| Completion time (pick→place) | ≤ 90 seg | ≤ 60 seg |
| Sim-to-real gap | Comportamiento similar sim/físico | < 15% diferencia en success rate |

## 2. Arquitectura y pipeline completo
El pipeline de la prueba tiene cinco fases en secuencia.
Las fases 3 y 4 (SDG y fine-tuning) son responsabilidad de Cosmos 3 y GR00T N respectivamente.
Las fases 1, 2 y 5 son de infraestructura e integración.

*(Figura 1 — Pipeline completo de la prueba. Azul: infraestructura/integración. Verde: Cosmos 3 / GR00T N.)*

### 2.1 Descripción de fases

| Fase | Responsable | Input | Output | Duración est. |
|---|---|---|---|---|
| 1. Setup entorno | Equipo infra | L40S + instancia | Docker + vLLM-Omni levantado | 1 día |
| 2. Captura escena | Equipo robótica | Espacio físico real | Scene USD + datasets base | 2 días |
| 3. SDG | Cosmos 3 Nano | Datasets + escena USD | 500+ pares (video,acción) | 2-3 días |
| 4. Fine-tuning | GR00T N1.6 | Dataset expandido | Modelo fine-tuneado G1 | 1-2 días |
| 5. Evaluación | Equipo completo | Modelo + G1 real/sim | Métricas + análisis | 2 días |

## 3. Setup del entorno

### 3.1 Infraestructura de compute
La generación de datos sintéticos y el fine-tuning corren en la instancia con L40S 48 GB.
El modelo de inferencia en el robot físico corre en el Jetson AGX Thor (o en la L40S en modo servidor remoto durante las pruebas iniciales).

| Componente | Spec | Rol en la prueba |
|---|---|---|
| GPU principal | NVIDIA L40S 48 GB GDDR6 | Cosmos 3 Nano SDG + Fine-tuning GR00T |
| CPU / RAM | Recomendado: 64 GB RAM | Layerwise-offload buffer de Cosmos 3 |
| Almacenamiento | ≥ 500 GB SSD | Modelos HF + datasets SDG + checkpoints |
| Robot compute | Unitree G1 onboard + Jetson Thor | Inferencia GR00T en tiempo real (fase final) |
| Red | LAN Gigabit / WiFi 5GHz | Comunicación ROS2 ↔ G1 |
| Simulación | Isaac Sim 4.x (GPU recomendado) | Digital Twin + evaluación en sim |

### 3.2 Instalación de Cosmos 3 Nano
Comando para levantar el servidor de inferencia de Cosmos 3 Nano en la L40S con layerwise-offload:

```bash
# 1. Descargar assets de ejemplo
pip install -U "huggingface_hub[cli]"
hf download nvidia/Cosmos3-Nano assets/ --local-dir Cosmos3-Nano

# 2. Levantar servidor vLLM-Omni
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v "$(pwd):/workspace" -p 8000:8000 --ipc=host vllm/vllm-omni:cosmos3 vllm serve nvidia/Cosmos3-Nano --omni --model-class-name Cosmos3OmniDiffusersPipeline --allowed-local-media-path / --enable-layerwise-offload --port 8000
```

El primer arranque descarga Cosmos3-Nano (~32 GB). Los tiempos de generación de video en L40S son más lentos que en H100/H200 (3-4x) pero completamente viables para SDG en batch.
Tener mínimo 64 GB de RAM del sistema para el layerwise-offload buffer.

### 3.3 Verificación del entorno
Comandos de verificación antes de iniciar el SDG:

```bash
# Verificar GPU visible
docker run --runtime nvidia --gpus all nvidia/cuda:12.0-base nvidia-smi

# Verificar que vLLM-Omni arrancó correctamente
curl http://localhost:8000/health
# Respuesta esperada: {"status": "ok"}

# Test rápido text-to-video
curl -X POST http://localhost:8000/v1/videos/sync -H 'Content-Type: application/json' -d '{"prompt": "A robot arm reaching for a water bottle on a table"}'
```

Si el servidor imprime 'Application startup complete.' en los logs, el modelo está listo.
El primer request de video tarda más por JIT compilation.

### 3.4 Paquetes ROS2 necesarios
* `ros-humble-nav2-bringup` (ya instalado con el stack de navegación)
* `unitree_ros2` — wrapper oficial de Unitree para ROS2 Humble
* `unitree_sdk2` — SDK de bajo nivel para control articular
* `groot_ros2_bridge` — nodo custom a desarrollar (ver Sección 5)
* `isaac-ros-common` — utilidades de integración ROS2 ↔ Isaac

## 4. Captura de escena y Digital Twin
Para que Cosmos 3 genere datos sintéticos físicamente plausibles, necesita una representación precisa del entorno de prueba.
Se captura la escena real con cámara 360° + LiDAR y se construye el Digital Twin en Isaac Sim.

### 4.1 Protocolo de captura

| Elemento | Herramienta | Especificación | Output |
|---|---|---|---|
| Captura visual 360° | Cámara 360° en selfie stick | Resolución mínima 4K · múltiples posiciones | Imágenes equirectangulares |
| Nube de puntos | LiDAR portátil (ej: GeoSLAM) | Densidad ≥ 100 pts/m² | Point cloud .ply / .las |
| Reconstrucción 3D | Gaussian Splatting / NeRF | Fusión visual + LiDAR | Mesh 3D / USD scene |
| Importación Isaac | Isaac Sim 4.x | Conversión USD con materiales PBR | Digital Twin operativo |

### 4.2 Configuración del espacio de prueba
El espacio de prueba debe cumplir los siguientes requisitos mínimos:
* Área despejada de al menos 3m × 3m para que el G1 pueda maniobrar con los brazos extendidos
* Mesa o superficie de origen a altura entre 0.6m y 0.9m (altura de manipulación óptima para G1)
* Superficie de destino (zona de placement) a 1.5m – 3m de distancia del origen
* Iluminación uniforme sin contraluz directa hacia las cámaras del G1
* Botella de agua: plástico translúcido 500ml, posición inicial definida y marcada
* Marcadores ArUco en puntos de referencia para calibración sim-to-real

El G1 usa cámaras RGB en la cabeza para la percepción de GR00T.
Durante las primeras pruebas, mantener la botella dentro del campo visual frontal del robot (±45° horizontal).

### 4.3 Setup del Digital Twin en Isaac Sim
* Importar la escena USD capturada al Isaac Sim Stage
* Importar el asset URDF del Unitree G1 (disponible en unitree_ros2)
* Configurar el Newton Physics Engine para simulación de contacto
* Agregar la botella de agua como rigid body con masa ~0.5 kg y fricción calibrada
* Configurar cámaras virtuales en las posiciones de las cámaras reales del G1
* Validar que la cinemática del robot en sim coincide con el comportamiento físico
* Exportar la escena como USD para usarla como condicionamiento en Cosmos 3

## 5. Generación de datos sintéticos (SDG) con Cosmos 3
Con el entorno levantado y la escena capturada, Cosmos 3 Nano genera los pares (video, acción) que se usarán para el fine-tuning de GR00T N1.6.
El flujo combina los datos reales de teleoperation con generación sintética aumentada.

*(Figura 2 — Pipeline de SDG. Inputs: datos de teleoperation real + digital twin. Cosmos 3 genera el dataset expandido.)*

### 5.1 Datos de entrada recomendados
Antes del SDG, capturar manualmente las siguientes demostraciones de teleoperation:
* 20 – 30 demostraciones de pick exitosas (botella en 5 posiciones distintas × 4-6 repeticiones)
* 10 – 15 demostraciones de carry (caminar con la botella sin soltarla, 2-3 metros)
* 10 – 15 demostraciones de place (depositar la botella en la zona de destino)
* 5 – 10 demostraciones fallidas intencionales (para que GR00T aprenda a recuperarse)

Cada demostración debe grabarse sincronizando:
* Video RGB de las cámaras del G1 (head camera)
* `joint_states.csv` a 50Hz durante toda la secuencia
* `imu.csv` y `force.csv` del gripper
* `events.log` con timestamps de START, GRASP, CARRY_START, PLACE, END

### 5.2 Configuración de los modos de generación
Se utilizan los tres modos de acción de Cosmos 3 para generar datasets complementarios:

| Modo | extra_params | Qué genera | Cantidad objetivo |
|---|---|---|---|
| Forward Dynamics (FDM) | `action_mode: forward`<br>`domain_name: bridge_orig_lerobot`<br>`raw_action_dim: 29` | Video del estado futuro dado acción — para que el robot imagine consecuencias | 200 variaciones |
| Inverse Dynamics (IDM) | `action_mode: inverse`<br>`domain_name: bridge_orig_lerobot`<br>`raw_action_dim: 29` | Trayectoria de acciones inferida de demostraciones reales — labeling automático | 150 trayectorias |
| Policy | `action_mode: policy`<br>`domain_name: bridge_orig_lerobot`<br>`action_chunk_size: 16` | Video futuro + acciones simultáneos dado imagen + instrucción texto | 200 pares completos |

### 5.3 Variaciones a generar
Para cada demostración base, Cosmos 3 genera variaciones automáticas usando domain randomization:
* Posición de la botella: 15 posiciones distintas dentro del área de alcance
* Iluminación: 4 condiciones (luz natural, artificial, contraluz difuso, sombras)
* Material/color de la botella: 3 variaciones (plástico transparente, azul, verde)
* Obstáculos parciales: 2 configuraciones de objetos próximos a la botella
* Fondo / superficie: 2 tipos de mesa (madera, metal)

Total dataset expandido objetivo: 500 – 800 pares (video, acción) listos para fine-tuning.

Usar el endpoint síncrono `POST /v1/videos/sync` para FDM (retorna solo video).
Para IDM y Policy usar el endpoint asíncrono `POST /v1/videos` y leer el resultado via `GET /v1/videos/{job_id}` al completarse, ya que incluyen el chunk de acción en el response.

### 5.4 Curación del dataset generado
Antes de pasar al fine-tuning, filtrar el dataset con los siguientes criterios:
* Descartar videos con inconsistencia temporal (morphing visible de la botella)
* Descartar trayectorias con joint values fuera del rango articular del G1
* Verificar alineación temporal entre video y vector de acción (±1 frame)
* Mantener al menos 30% de datos de demostraciones reales en el dataset final

## 6. Fine-tuning de GR00T N1.6

### 6.1 Configuración del entrenamiento
El fine-tuning parte de GR00T N1.6 como modelo base y se adapta a la tarea específica del G1 usando el dataset expandido generado por Cosmos 3. Se realiza en Isaac Lab sobre la instancia L40S.

| Parámetro | Valor recomendado | Notas |
|---|---|---|
| Modelo base | `nvidia/GR00T-N1-6` | Descargar desde HuggingFace |
| Tipo de entrenamiento | Supervised fine-tuning + action prediction | Imitation learning |
| Batch size | 4 – 8 | Ajustar según VRAM disponible en L40S |
| Learning rate | 1e-5 a 5e-5 | Empezar conservador, aumentar si no converge |
| Épocas | 20 – 50 | Validar en Isaac Sim cada 10 épocas |
| Action chunk size | 16 pasos | = 0.32 seg a 50Hz, coherente con IDM setup |
| Ratio datos sintéticos | 70% SDG / 30% real | Mantener datos reales para grounding |
| Checkpoints | Cada 10 épocas | Validar en sim antes de subir al robot físico |

### 6.2 Proceso de entrenamiento
* Descargar modelo base: `hf download nvidia/GR00T-N1-6` al directorio de trabajo
* Preparar el dataset en formato LeRobot (compatible con GR00T N): videos + JSON de acciones
* Configurar el training script de Isaac Lab con los parámetros de la tabla 6.1
* Iniciar el entrenamiento: `python train.py --config g1_pick_carry_place.yaml`
* Monitorear la loss de action prediction (objetivo: < 0.05 a las 20 épocas)
* Evaluar cada checkpoint en Isaac Sim usando Isaac Lab Arena
* Seleccionar el checkpoint con mejor task success rate en sim para deployment

GR00T N1.6 soporta post-training con el NVIDIA Physical AI Dataset de HuggingFace como base adicional.
Considerar mezclar ese dataset con el SDG propio para mejor generalización.

### 6.3 Validación en simulación (Isaac Lab Arena)
Antes de subir el modelo al robot físico, ejecutar la batería de evaluación en Isaac Sim:
* 50 episodios de pick en posiciones aleatorias dentro del espacio de entrenamiento
* 20 episodios de pick en posiciones fuera del rango de entrenamiento (test de generalización)
* 30 episodios de pick-carry-place end-to-end
* 10 episodios con perturbación (mover la botella levemente durante el carry)

Criterio de go/no-go para pasar al robot físico: task success rate ≥ 50% en los episodios de entrenamiento y ≥ 30% en los de generalización.

## 7. Integración ROS2 / Unitree SDK2
Esta es la fase de mayor desarrollo custom de la prueba.
GR00T N1.6 no tiene integración nativa con ROS2, por lo que se requiere construir el nodo bridge que conecta el output del modelo con el stack de control del G1.

*(Figura 3 — Arquitectura de integración. GR00T genera el vector de acción 29-D que el bridge traduce a comandos del G1.)*

### 7.1 Componentes del bridge

| Componente | Tecnología | Topic / Interface | Frecuencia |
|---|---|---|---|
| GR00T inference client | Python + requests | `POST /v1/videos/sync` | 50 Hz |
| Action publisher | rclpy node | pub: `/groot/action_output` (Float32MultiArray 29-D) | 50 Hz |
| Camera subscriber | rclpy node | sub: `/camera/rgb/image_raw` | 30 Hz |
| Joint command publisher | rclpy node | pub: `/joint_cmds` (JointState) | 50 Hz |
| NAV2 goal subscriber | rclpy node | sub: `/nav2/goal_reached` (Bool) | Event-driven |
| Árbitro de modo | rclpy node | Lógica: NAV=loco / MANIP=groot | 50 Hz |

### 7.2 Lógica del árbitro de control
El árbitro es el componente crítico que previene conflictos entre NAV2 (locomoción) y GR00T (manipulación).
La lógica de estados es la siguiente:

| Estado | Trigger de entrada | Control activo | Brazo | Trigger de salida |
|---|---|---|---|---|
| NAVIGATE | Inicio de tarea / waypoint asignado | NAV2 | Posición de transporte (hold) | Goal reached en waypoint de pick |
| PICK | Waypoint de origen alcanzado | GR00T N (Policy) | Activo — ejecuta pick | Botella detectada en gripper (force sensor) |
| CARRY | Grasp confirmado | NAV2 + GR00T arm stabilize | Hold dinámico durante marcha | Goal reached en waypoint de place |
| PLACE | Waypoint de destino alcanzado | GR00T N (Policy) | Activo — ejecuta place | Fuerza gripper < umbral (objeto soltado) |
| IDLE | Task completada o fallo | Ninguno | Posición home | Nueva tarea asignada |

### 7.3 Mapeo de acciones GR00T → G1 SDK
El vector de 29 dimensiones de GR00T debe mapearse a los comandos articulares del G1. Estructura de referencia:

```python
# Estructura del vector de acción 29-D para Unitree G1
# dims  0-2:  posición del torso (x, y, z)
# dims  3-5:  orientación del torso (roll, pitch, yaw)
# dims  6-11: brazo derecho (6 joints: shoulder×3, elbow, wrist×2)
# dims 12-17: brazo izquierdo (6 joints: idem)
# dims 18-20: gripper derecho (apertura, fuerza, rotación)
# dims 21-23: gripper izquierdo (idem)
# dims 24-28: cabeza (pan, tilt) + reservado
```

El mapeo exacto depende de la versión del URDF del G1 utilizado.
Verificar el archivo `joint_names.yaml` del paquete `unitree_ros2` y ajustar el bridge acordemente.
Durante las primeras pruebas, activar safety limits en SDK2 para joint velocity y torque.

### 7.4 Safety constraints para las pruebas
Configurar los siguientes límites de seguridad en el SDK2 antes de cualquier prueba física:
* Velocidad máxima de joints del brazo: 50% del límite nominal durante la fase de validación
* Torque máximo del gripper: calibrado para sostener 0.5 kg sin deformar la botella
* Emergency stop via topic ROS2: `/groot/emergency_stop` (Bool)
* Watchdog: si el bridge no publica en `/groot/action_output` por > 100ms, activar hold
* Zona de seguridad: bounding box de trabajo del brazo configurado en SDK2

## 8. Plan de evaluación y métricas

### 8.1 Estructura de la evaluación
La evaluación se realiza en dos fases: primero en simulación (Isaac Sim), luego en el robot físico.
Ambas usan el mismo protocolo para medir el sim-to-real gap.

| Fase | Entorno | Intentos | Variables controladas | Variables libres |
|---|---|---|---|---|
| A — Sim baseline | Isaac Sim | 50 episodios | Posición fija de botella | Nada: condiciones ideales |
| B — Sim variabilidad | Isaac Sim | 50 episodios | Posición aleatoria (espacio entrenamiento) | Posición botella |
| C — Sim generalización | Isaac Sim | 20 episodios | Posición fuera de entrenamiento | Posición + iluminación |
| D — Físico controlado | G1 real | 20 intentos | Posición fija · iluminación constante | Variaciones mecánicas reales |
| E — Físico variabilidad | G1 real | 20 intentos | Posición variable en rango entrenado | Posición + condiciones reales |

### 8.2 Métricas por fase de la tarea

| Métrica | Cómo medirla | Herramienta | Umbral mínimo |
|---|---|---|---|
| Task success rate (end-to-end) | Intentos exitosos / total intentos | Log manual + video | ≥ 50% (fases D-E) |
| Grasp success rate | Detección de fuerza en gripper + no-drop en 3s | force.csv + sensor | ≥ 70% |
| Drop rate (carry) | Detección de caída (force = 0) durante navegación | force.csv | ≤ 30% |
| Place accuracy | Distancia del placement al target (cm) | Marcador ArUco + camera | ≤ 15 cm |
| Completion time | Timestamp END - timestamp START | events.log | ≤ 90 seg |
| Recovery rate | Fallos recuperados / fallos totales | Log manual | ≥ 20% |
| Sim-to-real gap | \|(success_sim - success_real)\| / success_sim | Cálculo post-prueba | < 30% |

### 8.3 Protocolo de cada intento
* Posicionar la botella en la posición indicada por el protocolo de la sesión
* Asegurarse de que el G1 está en posición home y el gripper abierto
* Enviar el waypoint de origen via NAV2 (`/navigate_to_pose`)
* Activar grabación: `rosbag record -a -o intento_NNN`
* Enviar instrucción a GR00T: "toma la botella de agua"
* Observar ejecución sin intervención humana hasta completar o fallar
* Registrar resultado en la planilla de evaluación (éxito/fallo + motivo)
* Revisar `events.log` y `force.csv` para las métricas cuantitativas

Grabar todos los intentos en video externo (cámara fija) para análisis post-prueba.
El rosbag captura los datos de sensores pero el video externo permite identificar fallos de percepción vs. fallos de control.

### 8.4 Análisis de fallos
Clasificar cada fallo en una de las siguientes categorías para guiar las iteraciones de mejora:

| Categoría de fallo | Síntoma observable | Causa probable | Acción de mejora |
|---|---|---|---|
| Fallo de detección | El robot no encuentra la botella | Iluminación o posición fuera de distribución | Más variaciones en SDG |
| Fallo de agarre | Detecta pero no logra sujetar | Calibración de fuerza o ángulo de approach | Más demostraciones de grasp |
| Drop durante carry | Suelta la botella caminando | Control de estabilización durante marcha | Más datos de carry con perturbación |
| Fallo de placement | Suelta en posición incorrecta | Estimación de posición del destino | Más variaciones de zona de destino |
| Fallo de integración | Comportamiento errático o freeze | Bug en bridge ROS2 o timeout | Debug del bridge |

## 9. Checklist de preparación

### 9.1 Infraestructura
* L40S accesible y Docker con GPU runtime instalado
* Cosmos 3 Nano descargado y servidor vLLM-Omni levantado (GET /health = ok)
* Isaac Sim 4.x instalado y licencia activa
* Almacenamiento ≥ 500 GB disponible para modelos + datasets

### 9.2 Robot y navegación
* Unitree G1 encendido y conectado a ROS2 Humble via `unitree_ros2`
* NAV2 operativo: el G1 navega correctamente a los waypoints de prueba
* `joint_states`, `imu` y `force` topics publicando correctamente en ROS2
* Emergency stop (`/groot/emergency_stop`) testeado y funcional

### 9.3 Dataset y modelo
* Mínimo 20 demostraciones de teleoperation capturadas y sincronizadas
* Digital Twin validado en Isaac Sim (cinemática del G1 coherente)
* Dataset SDG generado (≥ 500 pares) y curado (filtrado de artefactos)
* GR00T N1.6 fine-tuneado y validado en Isaac Lab Arena (≥ 50% success rate en sim)
* `groot_ros2_bridge` implementado y testeado con el robot en modo simulado

### 9.4 Espacio de prueba
* Área de prueba de ≥ 3m × 3m despejada y demarcada
* Marcadores ArUco colocados en posiciones de referencia
* Cámara externa fija para grabación de video de los intentos
* Planilla de evaluación impresa o digital para registro de resultados
* Botella de agua de prueba disponible (500ml, plástico transparente)

## 10. Cronograma estimado

| Día | Actividad | Responsable | Entregable |
|---|---|---|---|
| Día 1 | Setup entorno: Docker, Cosmos 3 Nano, Isaac Sim, verificación GPU | Equipo infra | Servidor vLLM-Omni operativo + test de video OK |
| Día 2 | Captura de escena: 360° + LiDAR + reconstrucción 3D | Equipo robótica | Scene USD importada en Isaac Sim |
| Día 3 | Configuración Digital Twin + captura de demostraciones de teleoperation | Equipo robótica | ≥ 20 demos sincronizadas + Digital Twin validado |
| Día 4-5 | SDG con Cosmos 3 Nano: generación y curación del dataset expandido | Equipo IA | Dataset ≥ 500 pares curado y listo para training |
| Día 6 | Fine-tuning de GR00T N1.6 + evaluación en Isaac Lab Arena | Equipo IA | Modelo fine-tuneado con ≥ 50% success rate en sim |
| Día 7 | Desarrollo e integración del groot_ros2_bridge | Equipo robótica | Bridge testeado con robot en modo simulado |
| Día 8 | Pruebas físicas fases D y E: 40 intentos + registro de métricas | Equipo completo | Planilla de resultados + videos + rosbags |
| Día 9 | Análisis de resultados: métricas, clasificación de fallos, sim-to-real gap | Equipo completo | Informe de resultados + plan de mejoras |

El cronograma es una estimación. Los días 4-5 (SDG) pueden extenderse dependiendo de la velocidad de generación en L40S.
Priorizar la calidad del dataset sobre la velocidad: un dataset más pequeño y bien curado produce mejores resultados que uno grande con ruido.
