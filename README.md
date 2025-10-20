# CACIC2025-IAGyAI-Proyecto-Final
Proyecto final para el curso "IA Generativa y Agentes Inteligentes" de la Escuela Internacional del CACIC 2025.

Integrantes del grupo:

- Debernardis Kaamil
- Diaconchuk Hitomi
- Mella Rabinovich Fabian Ariel
- Pachano Blas

## Sobre el sistema multiagente:

En este repositorio se hace entrega del sistema multiagente solicitado como proyecto final basado en el ejemplo visto en clase de "Bruno", el mozo virtual (archivo original en /referencias) y se realizó con el objetivo de afianzar y demostrar el entendimiento de los contenidos vistos a lo largo del curso.

El sistema multiagente desarrollado consiste en un agente supervisor, un agente de RAG y un agente generador de reportes conectado con notion que colaboran para trabajar sobre un conjunto de registros varios de un proyecto de desarrollo de software (incidentes, minutas de reuniones, etc) en forma de archivos de texto plano.

<img width="550" height="372" alt="agent_workflow" src="https://github.com/user-attachments/assets/f443b6ef-9086-416f-94bf-221c9c350651" />

## Instrucciones para probar el sistema:

 - El sistema fue desarrollado en Python 3.13.7 y no fue probado en otras versiones.
 - Se deben instalar todas las dependencias listadas en el archivo requirements.txt con el comando "pip install -r requirements.txt" estando en el directorio raíz del proyecto (se recomienda crear un entorno virtual con " python -m venv .venv" para esto).
 - se deben configurar las variables de entorno en el archivo .env que incluyen las credenciales de Gemini, langsmith y Notion. Las credenciales de Langsmith y Notion (y la página web de Notion en la que se registran los informes) serán provistas por medio de los archivos que acompañan a esta entrega en la plataforma del curso.

## Extras:
 - Enlace a la página pública de Notion: https://enchanting-bard-465.notion.site/29125fd6a71e80678adcde78baddb666?v=29125fd6a71e8013acb4000cde6ce936&pvs=73
 - Enlace a la Project Run púnlica de Langsmith: https://smith.langchain.com/public/273eb0fd-2c70-4585-9937-cbd2d51b9843/r
