

# mcp
lista de mis herramientas mcp

## Configuración

Crea o actualiza `.env` con el asistente de configuración interactivo:

```bash
python setup_env.py
```

El script funciona en Windows, macOS y Linux. Cada integración es opcional,
los valores existentes se conservan por defecto y los valores secretos se ocultan mientras
se escribe. Ejecútalo nuevamente cada vez que quieras agregar o actualizar una integración.

No es necesario configurar cada integración. El servidor MCP aún se inicia
con un `.env` parcial o faltante; herramientas como GARUDA, Wikipedia, Google News,
clima de Open-Meteo, PDF y lectores de archivos locales permanecen disponibles. Una herramienta
que necesite credenciales faltantes devolverá un error de configuración únicamente cuando se invoque esa herramienta.

## Clima

Las herramientas de Open-Meteo proporcionan búsqueda de lugares, condiciones actuales, pronósticos por hora
y pronósticos diarios sin una clave API. Busca un lugar con
`search_weather_locations`, y luego pasa sus coordenadas a una herramienta de pronóstico.

## Conversión de divisas

Las herramientas Frankfurter listan monedas y proveedores, devuelven tasas actuales, históricas,
de múltiples monedas y específicas por proveedor, y convierten cantidades sin una clave
API. Los códigos de moneda usan notación de tres letras como `USD`, `IDR` y `EUR`.
`create_exchange_rate_chart` guarda un PNG en `downloads/charts` y también
lo devuelve como contenido de imagen MCP. Los artículos de arXiv se almacenan en `downloads/arxiv`.
`create_data_chart` proporciona la misma salida de alta resolución (DPI) para columnas seleccionadas en
archivos CSV o Excel, con etiquetas, anotaciones, estadísticas resumidas y notas de fuente.
Usa `inspect_data_file` primero para descubrir hojas, columnas, tipos de datos y filas de muestra.
Para conjuntos de datos ficticios, generados o pequeños, `create_inline_data_chart` acepta registros
directamente y devuelve el PNG sin crear un script o archivo de datos temporal.

## Búsqueda académica GARUDA

`search_garuda` refleja los filtros del formulario de búsqueda propio de GARUDA
(`https://garuda.kemdiktisaintek.go.id/documents`): pasa `search_field`
(`title`, `abstract`, `author` o `doi`) para restringir a qué campo debe coincidir la
consulta; usa `author` para búsquedas exactas por nombre de autor, ya que la búsqueda
por palabra clave simple no coincide de forma fiable con nombres de autores. Filtros adicionales:
`publisher` (nombre, mín. 3 caracteres), `pdf_only` (solo PDF descargable)
y `year_from`/`year_to` (rango de año de publicación).

## PlantUML

`render_plantuml` renderiza código fuente que contenga `@startuml` y `@enduml` como PNG
a través del Servidor oficial de PlantUML. Los resultados se guardan en
`downloads/plantuml` y se devuelven como contenido de imagen MCP para entrega en Telegram.

## Configuración de correo electrónico

Las herramientas de correo electrónico usan la configuración IMAP y SMTP de `.env`. Para Gmail, usa una
Contraseña de App de Google en lugar de la contraseña de la cuenta. Consulta `.env.example` para todas
las variables requeridas. Nunca hagas commit de `.env`.

## Configuración de Google Calendar y Drive

Coloca el archivo del cliente OAuth de Escritorio de Google en `credentials.json`, con la
API de Google Calendar y la API de Google Drive habilitadas en ese proyecto de Google Cloud.
Tanto `credentials.json` como `token.json` son ignorados por Git.

La autenticación es una llamada de herramienta en dos pasos, por lo que funciona igual ya sea que el servidor
se ejecute en tu portátil o en una máquina sin navegador alguno:

1. Llama a `google_auth_start`. Devuelve una URL de autorización.
2. Aprueba el acceso en un navegador; cualquier navegador, en cualquier dispositivo.
3. Llama a `google_auth_complete` con la URL donde aterrizas.

En un escritorio, los pasos 2 y 3 suelen ocurrir automáticamente: el navegador se abre
por ti y la redirección se intercepta en `http://localhost:8765` (establece
`GOOGLE_OAUTH_PORT` para cambiarlo), por lo que solo tienes que aprobar en el navegador.

En un servidor sin interfaz gráfica (headless), o cuando apruebas desde un teléfono, esa redirección no puede
alcanzarse y el navegador muestra un error de conexión. Esto es esperado; el
código de autorización está en la barra de direcciones. Copia la URL completa
(`http://localhost:8765/?code=...`) y pásala a `google_auth_complete`. También
acepta el valor `code` aislado si es más fácil de copiar.

Gana la ruta que termine primero, por lo que es seguro comenzar en el navegador
y recurrir a pegar la URL si es necesario. Los tokens se actualizan automáticamente después; solo repites este
proceso si el token es revocado o cambian los scopes solicitados.

El acceso a Drive usa el scope de solo lectura. Puede buscar archivos, inspeccionar metadatos,
leer Documentos y Hojas de cálculo como texto, y descargar o exportar archivos. A los usuarios existentes de Calendar
se les pedirá que autoricen nuevamente porque el token OAuth compartido ahora tiene
un scope adicional. Consulta `.env.example` para la configuración de rutas y zonas horarias.
