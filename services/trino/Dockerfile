FROM trinodb/trino:449

USER root

# Copy catalog configurations
COPY catalog/*.properties /etc/trino/catalog/

# Set proper permissions
RUN chown -R trino:trino /etc/trino/catalog && \
    chmod 644 /etc/trino/catalog/*.properties

USER trino