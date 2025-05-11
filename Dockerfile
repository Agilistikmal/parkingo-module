FROM python:latest
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 tesseract-ocr
RUN pip install -r requirements.txt --upgrade pip
EXPOSE 5000 
ENTRYPOINT [ "python" ] 
CMD [ "main.py" ] 