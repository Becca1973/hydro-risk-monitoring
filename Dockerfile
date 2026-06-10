FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-root

COPY . .

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "start.sh"]