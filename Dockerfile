FROM python:alpine3.14
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt --upgrade pip
EXPOSE 5000 
ENTRYPOINT [ "python" ] 
CMD [ "main.py" ] 