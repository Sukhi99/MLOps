## Use this version of python
FROM python:3.9.17-slim

RUN pip install poetry

# Set environment variable to prevent Poetry from creating a virtual environment
ENV POETRY_VIRTUALENVS_CREATE=false

## Creates directory "app" and cd's to that directory inside docker
WORKDIR /app
## copy the files "pyproject.toml" and "poetry.lock files" from local to inside docker's current directory
## cwd inside docker is /app -> because of WORKDIR /app
COPY ["pyproject.toml", "poetry.lock", "./"]

## Similar to pipenv install --system --deploy 
# RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
# RUN pip install -r requirements.txt
# Install only the main dependencies without creating a virtual environment
RUN poetry install --only=main --no-interaction --no-ansi

## Copying the model and predict.py script
COPY ["rs-churn-model.bin","predict.py", "./"]

## Exposing the port for talking to requests
EXPOSE 9696

## Entrypoint for predict function with gunicorn
# ENTRYPOINT ["gunicorn","--bind 0.0.0.0:9696","predict:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]

