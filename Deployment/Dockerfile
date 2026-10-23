# The bundled sklearn artifacts were serialized with scikit-learn 1.0.1.
# Python 3.8 is the closest practical runtime for this legacy model bundle.
FROM python:3.8-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip "setuptools<70" wheel && pip install -r requirements.txt
COPY . ./
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
