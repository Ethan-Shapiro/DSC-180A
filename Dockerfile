# syntax=docker/dockerfile:1.4
FROM 3.11.6-bookworm

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3"]
CMD ["run.py"]