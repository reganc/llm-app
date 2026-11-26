# The action gate's executor (see main.py): restarts listed apps, nothing else.
# Its code is baked in (no bind mount), so llm-api can't change it.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
