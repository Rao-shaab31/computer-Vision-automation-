import cv2
import numpy as np

def find_camera():
    backends = [
        ("Default", None),
        ("DSHOW",   cv2.CAP_DSHOW),
        ("MSMF",    cv2.CAP_MSMF),
    ]

    for backend_name, backend in backends:
        for i in range(5):
            try:
                if backend is None:
                    cap = cv2.VideoCapture(i)
                else:
                    cap = cv2.VideoCapture(i, backend)

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera found! Backend: {backend_name}, Index: {i}")
                        return cap
                cap.release()
            except Exception as e:
                print(f"Error trying {backend_name} index {i}: {e}")

    return None

# Find and open camera
cap = find_camera()

if cap is None:
    print("Error: No working camera found.")
    print("Please check:")
    print("  1. Camera is connected and not used by another app")
    print("  2. Windows camera privacy settings (Win+I > Privacy > Camera)")
    print("  3. Device Manager for driver issues")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges (HSV) — (lower, upper, box_color in BGR)
    colors = {
        "red":    ([0, 120, 70],    [10, 255, 255],  (0, 0, 255)),
        "green":  ([40, 40, 40],    [80, 255, 255],  (0, 255, 0)),
        "yellow": ([20, 100, 100],  [30, 255, 255],  (0, 255, 255)),
        "blue":   ([100, 150, 50],  [140, 255, 255], (255, 0, 0)),
        "orange": ([10, 100, 100],  [20, 255, 255],  (0, 165, 255)),
    }

    for name, (lower, upper, box_color) in colors.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)

        # Create mask
        mask = cv2.inRange(hsv, lower_np, upper_np)

        # Reduce noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)

            # Ignore very small areas (noise)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # Show area size
                area = cv2.contourArea(largest)
                cv2.putText(frame, f"Area: {int(area)}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

    # Show FPS on screen
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show instructions
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show output
    cv2.imshow("Color Tracker", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera released. Program ended.")