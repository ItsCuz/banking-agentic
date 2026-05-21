import sys
import uvicorn

# Reconfigure stdout/stderr to support Vietnamese text and emojis in Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass # Python 3.7+ supports reconfigure

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)