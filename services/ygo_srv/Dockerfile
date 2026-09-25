FROM python:3.12-slim

WORKDIR /app

COPY services/tcg_srvs/ygo_srv/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/tcg_srvs/ygo_srv/ .

CMD ["uvicorn", "ygo_srv:app", "--host", "0.0.0.0", "--port", "8000"]