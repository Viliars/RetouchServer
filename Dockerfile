FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ADD requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

ADD src /app
ADD retouch.pt /app/retouch.pt

WORKDIR /app

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2600", "--workers", "4"]