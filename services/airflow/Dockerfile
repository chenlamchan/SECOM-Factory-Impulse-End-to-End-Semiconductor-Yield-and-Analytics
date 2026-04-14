FROM apache/airflow:2.8.1-python3.11

USER root

# 1. Install System Dependencies (Java, Build Tools, wget)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        build-essential \
        g++ \
        libre2-dev \
        wget \
        procps && \
    apt-get autoremove -yqq --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Spark Binaries (while still root)
ENV SPARK_HOME=/opt/spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.3-bin-hadoop3.tgz && \
    mv spark-3.5.3-bin-hadoop3 $SPARK_HOME && \
    rm spark-3.5.3-bin-hadoop3.tgz && \
    # Give the airflow user ownership of the spark folder
    chown -R airflow:root $SPARK_HOME

# Switch back to the airflow user for Python installations
USER airflow

# 3. Install Python requirements
COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/home/airflow/.cache/pip,sharing=locked \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$SPARK_HOME/bin:$PATH"