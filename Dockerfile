FROM python:alpine3.13
COPY . /app
WORKDIR /app
RUN pip install onnxruntime
RUN pip install -r requirements.txt --upgrade pip
EXPOSE 5000 
ENTRYPOINT [ "python" ] 
CMD [ "main.py" ] 