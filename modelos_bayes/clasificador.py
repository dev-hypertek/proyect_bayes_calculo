from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import re
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import sys
import os

# Añadir la ruta del directorio padre al path de forma absoluta
sistema_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, sistema_path)

# Importar la configuración de Flask
try:
    import config_flask  # Importar el módulo de configuración
    print("Módulo config_flask importado correctamente")
except ImportError as e:
    print(f"Error al importar config_flask: {e}")
    print(f"sys.path: {sys.path}")
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Archivos en directorio raíz: {os.listdir(sistema_path)}")
    raise

# Obtener la ruta base del proyecto
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(base_dir, 'frontend')

# Configuración de la aplicación Flask
app = Flask(__name__, static_folder=frontend_path, static_url_path='')
app = config_flask.configure_app(app)  # Configurar la aplicación

# Configurar CORS correctamente para permitir solicitudes desde cualquier origen
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})


#########################################################################
########              DATASET Y CARGUE DE DATOS             #############
#########################################################################
# Categorías posibles
CATEGORIAS = [ 
    "Admisión, Registro y Control",
    "Aplazamiento",
    "Apoyo Financiero",
    "Atención al cliente CDE",
    "Bienestar, Deporte y Cultura",
    "Ceremonia de Grados",
    "Dirección de Educación Virtual",
    "Facultad de Ciencias Sociales y de la Educación",
    "Facultad de Ingeniería",
    "Facultad Escuela de Negocios",
    "Gestión comercial nuevos",
    "Internacionalización",
    "Personal externo e interno",
    "PQRSF",
    "Salud y Prevención",
    "Tecnología"
]

# Palabras clave relacionadas con cada categoría (simple para el demo)
PALABRAS_CLAVE = {
    "Admisión, Registro y Control": ["admisión", "registro", "certificado", "notas", "calificaciones", "matrícula", "documentos", "inscripción"],
    "Aplazamiento": ["aplazamiento", "aplazar", "posponer", "suspender", "semestre", "período"],
    "Apoyo Financiero": ["beca", "descuento", "financiación", "pago", "préstamo", "crédito", "factura", "recibo", "financiero"],
    "Atención al cliente CDE": ["atención", "cliente", "cde", "centro", "consulta", "ayuda"],
    "Bienestar, Deporte y Cultura": ["bienestar", "deporte", "cultura", "gimnasio", "evento", "actividad", "cultural", "recreación", "torneo"],
    "Ceremonia de Grados": ["ceremonia", "grado", "graduación", "diploma", "toga", "birrete"],
    "Dirección de Educación Virtual": ["virtual", "online", "plataforma", "aula", "moodle", "edv", "educación virtual", "remoto"],
    "Facultad de Ciencias Sociales y de la Educación": ["ciencias sociales", "educación", "pedagogía", "sociología", "psicología", "humanidades"],
    "Facultad de Ingeniería": ["ingeniería", "sistemas", "software", "programación", "técnico", "tecnológico", "civil", "industrial"],
    "Facultad Escuela de Negocios": ["negocios", "administración", "contaduría", "economía", "finanzas", "marketing", "empresarial"],
    "Gestión comercial nuevos": ["comercial", "ventas", "promoción", "prospecto", "nuevo estudiante", "información carrera"],
    "Internacionalización": ["internacional", "intercambio", "movilidad", "extranjero", "convenio", "global"],
    "Personal externo e interno": ["profesor", "docente", "administrativo", "empleado", "contrato", "personal", "laboral"],
    "PQRSF": ["petición", "queja", "reclamo", "sugerencia", "felicitación", "inconformidad", "problema", "pqrs"],
    "Salud y Prevención": ["salud", "enfermedad", "médico", "psicológico", "prevención", "emergencia", "accidente", "incapacidad"],
    "Tecnología": ["computador", "software", "correo", "contraseña", "sistema", "acceso", "wifi", "internet", "tecnológico", "app"]
}




#########################################################################
########          MODELO DE CLASIFICACION BAYESIANA         #############
#########################################################################

# Ruta al modelo entrenado y vectorizador
MODELO_PATH = 'modelos_bayes/modelo_clasificador.pkl'
VECTORIZADOR_PATH = 'modelos_bayes/vectorizador.pkl'

# Cargar o entrenar el modelo
def cargar_o_entrenar_modelo():
    """Carga el modelo preentrenado si existe, o entrena uno nuevo si no"""
    global vectorizador, modelo
    
    # Verificar si existe el modelo guardado
    if os.path.exists(MODELO_PATH) and os.path.exists(VECTORIZADOR_PATH):
        try:
            # Cargar modelo y vectorizador preentrenados
            with open(MODELO_PATH, 'rb') as f:
                modelo = pickle.load(f)
            with open(VECTORIZADOR_PATH, 'rb') as f:
                vectorizador = pickle.load(f)
            print("Modelo cargado correctamente desde archivos.")
            return
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
    
    # Si no hay modelo guardado o hubo un error, entrenar uno nuevo
    print("Entrenando nuevo modelo...")
    
    try:
        # Cargar el dataset
        ruta_dataset = "modelos_bayes/Data (2).csv"
        
        # Primero, verificar el contenido del archivo
        try:
            with open(ruta_dataset, 'r', encoding='utf-8') as f:
                primeras_lineas = [next(f) for _ in range(10)]
                print("Contenido de las primeras líneas del CSV:")
                for i, linea in enumerate(primeras_lineas):
                    print(f"Línea {i+1}: {linea.strip()}")
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
        
        try:
            # Intentar cargar con detección automática de delimitador
            try:
                # Para pandas >= 1.3.0
                df = pd.read_csv(ruta_dataset, encoding='utf-8', engine='python', on_bad_lines='skip')
            except TypeError:
                # Para pandas < 1.3.0
                df = pd.read_csv(ruta_dataset, encoding='utf-8', engine='python', error_bad_lines=False, warn_bad_lines=True)
        except Exception as e1:
            try:
                # Segundo intento con parámetros específicos
                df = pd.read_csv(ruta_dataset, encoding='latin-1', sep=',', on_bad_lines='skip')
            except TypeError:
                # Para pandas < 1.3.0
                df = pd.read_csv(ruta_dataset, encoding='latin-1', sep=',', error_bad_lines=False)
            except Exception as e2:
                print(f"Error al cargar el dataset: {e1} / {e2}")
                print("Creando un modelo básico de respaldo...")
                
                # Crear un dataset básico para entrenamiento
                datos_basicos = {
                    "Departamento": [
                        "Administrativo", "Administrativo", "Administrativo", "Administrativo", "Administrativo",
                        "Apoyo financiero", "Apoyo financiero", "Apoyo financiero", "Apoyo financiero", "Apoyo financiero",
                        "Registro y control", "Registro y control", "Registro y control", "Registro y control", "Registro y control"
                    ],
                    "Solicitud": [
                        "Solicito un certificado de notas", "Requiero una constancia de estudio", 
                        "Necesito una copia de mi diploma", "Por favor emitir certificado", "Certificación académica",
                        "Solicito beca", "Necesito financiación", "Pago fraccionado", "Descuento por convenio", "Soporte de pago",
                        "Cambio de jornada", "Cancelar materia", "Homologación de asignaturas", "Aplazamiento de semestre", "Reactivar matrícula"
                    ]
                }
                df = pd.DataFrame(datos_basicos)
        
        # Preparar datos para entrenamiento
        X = df['Solicitud']
        y = df['Departamento']
        
        # Crear y entrenar el vectorizador
        vectorizador = CountVectorizer()
        X_vectorizado = vectorizador.fit_transform(X)
        
        # Crear y entrenar el modelo
        modelo = MultinomialNB()
        modelo.fit(X_vectorizado, y)
        
        # Guardar el modelo y vectorizador para uso futuro
        with open(MODELO_PATH, 'wb') as f:
            pickle.dump(modelo, f)
        with open(VECTORIZADOR_PATH, 'wb') as f:
            pickle.dump(vectorizador, f)
        
        print("Modelo entrenado y guardado correctamente.")
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        # Si falla el entrenamiento, usaremos el clasificador basado en palabras clave

# Inicializar variables globales para el modelo y vectorizador
vectorizador = None
modelo = None

# Función para clasificar texto nuevo
def clasificar_texto(texto):
    """Clasifica el texto usando el modelo entrenado o palabras clave como fallback"""
    # Si el modelo no está cargado, intentar cargarlo
    if modelo is None or vectorizador is None:
        cargar_o_entrenar_modelo()
    
    # Si el modelo se cargó correctamente, usarlo para clasificar
    if modelo is not None and vectorizador is not None:
        try:
            # Preprocesar el texto
            texto_limpio = texto.lower()
            texto_limpio = re.sub(r'[^\w\s]', '', texto_limpio)
            
            # Vectorizar el texto
            texto_vectorizado = vectorizador.transform([texto_limpio])
            
            # Predecir la categoría
            categoria = modelo.predict(texto_vectorizado)[0]
            return categoria
        except Exception as e:
            print(f"Error al clasificar con el modelo: {e}")
            # Si falla, usar el método de palabras clave como respaldo
    
    # Método de respaldo basado en palabras clave
    print("Usando clasificación por palabras clave como respaldo")
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.split()
    
    # Calcular puntajes por categoría basados en coincidencias de palabras clave
    puntajes = {}
    for categoria, keywords in PALABRAS_CLAVE.items():
        puntajes[categoria] = 0
        for palabra in palabras:
            if palabra in keywords or any(keyword in texto for keyword in keywords):
                puntajes[categoria] += 1
    
    # Si no hay coincidencias, asignar a PQRSF por defecto
    if max(puntajes.values()) == 0:
        return "PQRSF"
    
    # Devolver la categoría con mayor puntaje
    return max(puntajes.items(), key=lambda x: x[1])[0]




#########################################################################
########          GUARDA REGISTROS Y ENVIA CORREO           #############
#########################################################################
def guardar_registro(texto, categoria):
    """
    Guarda el registro en un archivo de texto
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("modelos_bayes/tickets.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {categoria}: {texto}\n")
        f.write("-"*80 + "\n")

def enviar_correo(texto, categoria):
    """
    Envía una notificación por correo electrónico
    """
    try:
        # Configuración del servidor de correo (esto es para Gmail, ajustar según sea necesario)
        sender_email = "tu_correo@gmail.com"  # Reemplazar con tu correo
        sender_password = "tu_contraseña"     # Reemplazar con tu contraseña o token de app
        receiver_email = "bbleon@ucompensar.edu.co"
        
        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"Nueva solicitud clasificada: {categoria}"
        
        # Cuerpo del mensaje
        body = f"""
        Se ha recibido una nueva solicitud:
        
        Categoría: {categoria}
        Descripción: {texto}
        
        Fecha y hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        Este es un mensaje automático, no responder.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Conexión al servidor SMTP
        # Nota: Esto es un ejemplo y debe modificarse según el servidor de correo real
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Error al enviar correo: {e}")
        return False




#########################################################################
########            COREXION CON FRONTEND                   #############
#########################################################################
@app.route('/clasificar', methods=['POST', 'OPTIONS'])
def clasificar_solicitud():
    # Manejar la solicitud OPTIONS para CORS
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    data = request.json
    texto = data.get('descripcion', '')
    
    if not texto:
        return jsonify({'error': 'No se proporcionó texto para clasificar'}), 400
    
    # Clasificar el texto
    categoria = clasificar_texto(texto)
    
    # Guardar en archivo
    guardar_registro(texto, categoria)
    
    # Intentar enviar correo (comentado para el demo)
    # enviar_correo(texto, categoria)
    
    response = jsonify({
        'success': True,
        'area': categoria
    })
    
    # Añadir encabezados CORS manualmente
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Cargar el modelo al iniciar la aplicación
cargar_o_entrenar_modelo()

if __name__ == '__main__':
    print("\n===================================================")
    print("      Sistema de Gestión de Tickets UCompensar")
    print("===================================================\n")
    # Información de depuración para verificar las rutas
    print(f"Ruta de archivos estáticos: {app.static_folder}")
    print(f"URL path de archivos estáticos: {app.static_url_path}")
    print(f"Verificando archivos estáticos:")
    print(f"  index.html: {os.path.exists(os.path.join(app.static_folder, 'index.html'))}")
    print(f"  crear_caso.html: {os.path.exists(os.path.join(app.static_folder, 'crear_caso.html'))}")
    print(f"  logo_compensar.png: {os.path.exists(os.path.join(app.static_folder, 'logo_compensar.png'))}")
    
    print("Servidor iniciado en http://localhost:5050")
    print("Presiona CTRL+C para detener el servidor\n")
    app.run(debug=True, host='0.0.0.0', port=5050)