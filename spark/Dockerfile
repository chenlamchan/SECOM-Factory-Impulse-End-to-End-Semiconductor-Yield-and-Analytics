# ----------------------------------------------------------------------------
# Spark 3.5 Image with Direct JAR Downloads from Maven Central
# ----------------------------------------------------------------------------
FROM apache/spark:3.5.3-python3 as final

USER root
WORKDIR /opt/spark/work

RUN --mount=type=cache,target=/var/cacahe/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Fix home directory
RUN mkdir -p /home/spark && \
    sed -i 's|/nonexistent|/home/spark|g' /etc/passwd && \
    chown -R 185:0 /home/spark /opt/spark/work && \
    chmod -R 755 /home/spark

# ============================================================================
# Download JARs directly from Maven Central
# ============================================================================

ARG ICEBERG_VERSION=1.7.0
ARG POSTGRES_VERSION=42.6.0
ARG AWS_SDK_BUNDLE_VERSION=2.25.33
ARG HADOOP_VERSION=3.3.4
ARG AWS_JAVA_SDK_VERSION=1.12.262

RUN --mount=type=cache,target=/tmp/jar-cache,sharing=locked \
    cd /opt/spark/jars && \
    for url in \
        "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar" \
        "https://repo1.maven.org/maven2/org/postgresql/postgresql/${POSTGRES_VERSION}/postgresql-${POSTGRES_VERSION}.jar" \
        "https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/${AWS_SDK_BUNDLE_VERSION}/bundle-${AWS_SDK_BUNDLE_VERSION}.jar" \
        "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" \
        "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_JAVA_SDK_VERSION}/aws-java-sdk-bundle-${AWS_JAVA_SDK_VERSION}.jar" \
    ; do \
        filename=$(basename "$url"); \
        if [ ! -f "/tmp/jar-cache/$filename" ]; then \
            wget -q -P /tmp/jar-cache/ "$url"; \
        fi; \
        cp "/tmp/jar-cache/$filename" .; \
    done && \
    echo "=== Downloaded JARs ===" && \
    ls -lh /opt/spark/jars/ | grep -E "(iceberg|postgresql|aws|hadoop)"

# Copy application code
COPY spark_jobs /opt/spark/src

# Set permissions
RUN chown -R 185:0 /opt/spark/src /opt/spark/jars && \
    chmod -R a+rX /opt/spark/src

USER 185
ENV HOME=/home/spark