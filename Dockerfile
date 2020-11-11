FROM ubuntu:18.04
WORKDIR /app
COPY requirements.txt .
RUN apt update && apt install -y python3-pip libgl1-mesa-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY . .
EXPOSE 7000
RUN python3 preload.py
CMD ["sh", "-c", "gunicorn seq2seq_api:app -b 0.0.0.0:7000"]