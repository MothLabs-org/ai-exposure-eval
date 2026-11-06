FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer-cached separately from source code).
# constraints.txt pins exact versions, so a rebuild can't silently upgrade
# Streamlit, pandas or anything else.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy source
COPY . .

# Default port — override with PORT env var or docker-compose
ARG PORT=7860
ENV PORT=${PORT}

EXPOSE ${PORT}

# python:3.13-slim ships without curl, so probe with the stdlib instead.
# Streamlit moves its health endpoint under STREAMLIT_SERVER_BASE_URL_PATH
# when one is set (docker-compose.yml sets it; the image default is the root).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request;p=os.environ.get('STREAMLIT_SERVER_BASE_URL_PATH','').strip('/');urllib.request.urlopen('http://localhost:'+os.environ['PORT']+'/'+(p+'/' if p else '')+'_stcore/health')" || exit 1

# Run Streamlit — results dirs are expected to be mounted at runtime
CMD streamlit run dashboard.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
