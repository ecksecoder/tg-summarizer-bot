FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 botuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and configure permissions
COPY --chown=botuser:botuser . .
RUN chown -R botuser:botuser /app

# Switch to non-root user
USER botuser

# Start application
CMD ["python", "bot.py"]
