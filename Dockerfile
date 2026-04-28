# Stage 1 — Builder
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2 — Final
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY noshow_iq/ ./noshow_iq/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "8000"]