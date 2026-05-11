import subprocess
import psycopg2
import time

conn = None
for i in range(10):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="user",
            password="password",
            host="db"
        )
        break
    except psycopg2.OperationalError:
        time.sleep(2)

if not conn:
    print("Could not connect to DB")
    exit(1)

result = subprocess.run(['./blas_bench'], capture_output=True, text=True)
output = result.stdout.strip()

if "BLAS_TEST" in output:
    time_val = float(output.split()[1])
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO test_results (test_name, execution_time) VALUES (%s, %s)",
        ("Scalar_Multiply", time_val)
    )
    conn.commit()
    cur.close()
    conn.close()
    print("Results recorded to DB")
