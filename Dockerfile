FROM python:3.12.3-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app
# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./
COPY . /app
# Install the dependencies in the system's Python environment
RUN pipenv install --deploy --system

CMD ["python3", "app.py"]