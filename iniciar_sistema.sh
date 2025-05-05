#!/bin/bash

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================================"
echo -e "         Sistema de Gestión de Tickets UCompensar"
echo -e "========================================================${NC}"
echo ""

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creando entorno virtual...${NC}"
    python3 -m venv venv || python -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error al crear el entorno virtual. Verifique que Python esté instalado.${NC}"
        read -p "Presiona Enter para salir..."
        exit 1
    fi
fi

# Activar el entorno virtual
echo -e "${BLUE}Activando entorno virtual...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Error al activar el entorno virtual.${NC}"
    read -p "Presiona Enter para salir..."
    exit 1
fi

# Instalar dependencias
echo -e "${BLUE}Instalando dependencias...${NC}"
pip install -r requirements.txt

# Verificar y reparar el archivo CSV
echo -e "${BLUE}Verificando el archivo CSV de datos...${NC}"
# No se necesita verificar el CSV, ya se maneja en clasificador.py

# Iniciar la aplicación
echo -e "${GREEN}¡Todo listo! Iniciando la aplicación...${NC}"
echo -e "${GREEN}========================================================"
echo -e "  La aplicación se abrirá en tu navegador automáticamente"
echo -e "========================================================${NC}"

# Abrir el navegador después de 2 segundos
(sleep 2 && python -c "import webbrowser; webbrowser.open('http://localhost:5050')") &

# Iniciar el servidor
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
python modelos_bayes/clasificador.py
