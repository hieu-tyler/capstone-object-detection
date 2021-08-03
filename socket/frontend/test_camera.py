import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break
    else:
        print(frame.shape)
        cv2.imshow('Video Output', frame)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing camera
camera.release()
# Destroying all opened OpenCV windows
cv2.destroyAllWindows()