FROM apache/airflow:2.8.1

# Install Python requirements
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt && rm requirements.txt