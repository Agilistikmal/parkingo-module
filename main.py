import plate_scanner
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        plate_number = plate_scanner.scan(frame)
        if plate_number != None and plate_number != "":
            print(plate_number)
        
        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()