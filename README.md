# Sistema de Gestión de Tickets con Clasificación Automática - UCompensar

Sistema demo para gestión de tickets con clasificación bayesiana automática para UCompensar.

## Características

- Página de bienvenida (index.html)
- Formulario simplificado de creación de tickets
- Clasificador bayesiano automático
- Almacenamiento en archivo de texto
- Notificación por correo electrónico

## Requisitos

- Python 3.6+
- Dependencias en requirements.txt

## Instalación y Ejecución

### Windows
```
> iniciar_sistema.bat
```

### macOS/Linux
```
$ chmod +x iniciar_sistema.sh
$ ./iniciar_sistema.sh
```

El sistema se iniciará en http://localhost:5050

## Estructura del Proyecto

```
proyecto_bayes/
├── frontend/                    # Archivos de interfaz de usuario
│   ├── index.html              # Página de bienvenida
│   ├── crear_caso.html         # Formulario de tickets
│   └── logo_compensar.png      # Logo institucional
├── modelos_bayes/               # Modelos y backend
│   ├── clasificador.py         # Script principal con algoritmo bayesiano
│   ├── Data (2).csv            # Datos de entrenamiento
│   ├── modelo_clasificador.pkl # Modelo serializado
│   └── vectorizador.pkl        # Vectorizador serializado
├── config_flask.py              # Configuración de Flask
├── requirements.txt             # Dependencias
├── iniciar_sistema.bat          # Script de inicio para Windows
└── iniciar_sistema.sh           # Script de inicio para macOS/Linux
```

## Clasificación de Tickets

El sistema clasifica automáticamente las solicitudes en estas categorías:
- Admisión, Registro y Control
- Aplazamiento
- Apoyo Financiero
- Atención al cliente CDE
- Bienestar, Deporte y Cultura
- Ceremonia de Grados
- Entre otras

## Tecnologías

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- Machine Learning: Naive Bayes (scikit-learn)

---

Desarrollado por Brandow Brusly Leon Rodriguez - 2025
