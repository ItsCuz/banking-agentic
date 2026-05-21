FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /src

COPY requirements-api.txt /src/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /src/requirements-api.txt

COPY app /src/app
COPY configs /src/configs
COPY run.py /src/run.py

EXPOSE 8000

CMD ["python", "run.py"]
