FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    g++ \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install psycopg2-binary
WORKDIR /app
COPY main.cpp record_results.py /app/
RUN g++ -O3 main.cpp -o blas_bench
CMD ["python", "record_results.py"]