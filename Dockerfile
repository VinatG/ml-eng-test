# Building environment over python 3.9
FROM python:3.9-slim
WORKDIR /app
# Copying all the directory contents into docker container
COPY . /app 
# Installing all the necessary python packages using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 
# Using port 3000 as mentioned by the authors of this assesment
EXPOSE 3000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Running FastAPI server using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--limit-concurrency", "50", "--limit-max-requests", "1000"]



