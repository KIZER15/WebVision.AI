FROM python:3.10

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    chromium-driver \
    chromium \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    wget \
    ca-certificates \
    fonts-liberation \
    xdg-utils \
    --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome
ENV CHROME_BIN="/usr/bin/chromium"
ENV CHROMEDRIVER_PATH="/usr/bin/chromedriver"

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

