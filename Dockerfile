# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7-slim-buster

# Install pip requirements
COPY requirements.txt .
# libraries for opencv
RUN python -m pip install -r requirements.txt

RUN apt-get update

WORKDIR /app
COPY . /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
EXPOSE 8000
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]