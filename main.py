from flask import Flask, request
from dotenv import load_dotenv
import os
import plate_scanner
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

@app.post("/scanner")
def scanner():
    x_api_key = request.headers.get("X-API-KEY")
    x_mac_address = request.headers.get("X-MAC-ADDRESS")
    api_key = os.environ.get("API_KEY")

    if x_api_key == None and x_mac_address == None:
        return {
            "error": "Required header X-API-KEY and X-MAC-ADDRESS",
            "data": None
        }, 400
    
    if api_key != x_api_key:
        return {
            "error": "Invalid X-API-KEY",
            "data": None
        }, 401
    
    image = request.files.get("image")

    if image == None or image.filename == "":
        return {
            "error": "Required image files body",
            "data": None
        }, 400
    
    try:
        image_bytes = image.read()
        image_stream = io.BytesIO(image_bytes)
        with Image.open(image) as img:
            img.verify()

            image_stream.seek(0)
            img = Image.open(image_stream)
        
            npimg = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            plate_number = plate_scanner.scan(frame=frame)
            
            return {
                    "error": None,
                    "data": plate_number
                }, 200
    except (IOError, SyntaxError):
        return {
                "error": "Uploaded file is not an image",
                "data": None
            }, 400

if __name__ == "__main__":
    load_dotenv()
    app.run()