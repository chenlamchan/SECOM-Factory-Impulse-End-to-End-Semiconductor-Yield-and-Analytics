# ----------------------------------------------------------------------------
# Spark 3.5 Image with Direct JAR Downloads from Maven Central
# ----------------------------------------------------------------------------
FROM apache/spark:3.5.3-python3 as final

USER root
WORKDIR /opt/spark/work

# Install wget for downloading JARs
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Fix home directory
RUN mkdir -p /home/spark && \
    sed -i 's|/nonexistent|/home/spark|g' /etc/passwd && \
    chown -R 185:0 /home/spark /opt/spark/work && \
    chmod -R 755 /home/spark

# ============================================================================
# Download JARs directly from Maven Central
# ============================================================================

# Iceberg runtime
RUN cd /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.7.0/iceberg-spark-runtime-3.5_2.12-1.7.0.jar

# Postgres JDBC (For Iceberg Catalog)
RUN cd /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/postgresql/postgresql/42.6.0/postgresql-42.6.0.jar

# AWS SDK Bundle for S3 (MinIO)
RUN cd /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.25.33/bundle-2.25.33.jar

# Hadoop AWS
RUN cd /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Verify all JARs were downloaded
RUN echo "=== Downloaded JARs ===" && \
    ls -lh /opt/spark/jars/*.jar | grep -E "(iceberg|postgresql|aws|hadoop)" && \
    echo "=== Total JAR count ===" && \
    ls -1 /opt/spark/jars/*.jar | wc -l

# Copy application code
COPY spark_jobs /opt/spark/src

# Set permissions
RUN chown -R 185:0 /opt/spark/src /opt/spark/jars && \
    chmod -R a+rX /opt/spark/src

USER 185
ENV HOME=/home/spark