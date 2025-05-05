"""
Archivo de configuración para Flask.
Este script modifica la configuración de Flask para servir archivos estáticos.
"""

import os
from flask import send_from_directory, send_file

def configure_app(app):
    """Configura la aplicación Flask para servir archivos estáticos correctamente."""
    # Flask ya debería tener configurado el static_folder desde la inicialización
    
    # Añadir rutas para servir los archivos HTML
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
    @app.route('/crear_caso.html')
    def crear_caso():
        return app.send_static_file('crear_caso.html')

    # Asegurar que los archivos estáticos se sirvan correctamente
    @app.route('/<path:filename>')
    def serve_static(filename):
        return send_from_directory(app.static_folder, filename)
    
    # Desactivar strict_slashes para permitir URLs con o sin barra final
    app.url_map.strict_slashes = False
    
    # Configurar cabeceras CORS para todos los endpoints
    @app.after_request
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    return app
