import cv2
import numpy as np
import os

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_dir = "face_data"
    faces = []
    labels = []
    label_dict = {}
    current_id = 0

    for employee_name in os.listdir(face_dir):
        employee_path = os.path.join(face_dir, employee_name)
        if not os.path.isdir(employee_path):
            continue

        label_dict[current_id] = employee_name
        for image_name in os.listdir(employee_path):
            image_path = os.path.join(employee_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(current_id)
        current_id += 1

    if not faces:
        print("No face data available to train.")
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")

    # Convert integer keys to strings for np.savez
    label_dict_str = {str(k): v for k, v in label_dict.items()}
    np.savez("label_dict.npz", **label_dict_str)

    print("Model trained successfully!")
    return True

if __name__ == "__main__":
    train_model()