import cv2
import numpy as np
import os
import mediapipe as mp
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy import stats

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'action.h5')
DATA_PATH = os.path.join(BASE_DIR, 'MP_Data')

# Define actions
actions = np.array(['call', 'police', 'me', 'help', 'iloveyou', 'hello', 'thankyou', 'bye', 'danger'])

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Pushover notification setup
api_token = 'av48uwk6ic2oheyyjf44phrt47q5dz'
user_key = 'ugw9227sfr8ih35wrejgaku7533sbe'
target_sequences = [['call', 'police'], ['help', 'me'], ['danger', 'call', 'police']]

# Colors for visualization
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 0),
          (255, 255, 0), (255, 0, 255), (128, 128, 128), (0, 128, 0)]

def mediapipe_detection(image, model):
    """
    Detect landmarks in an image using MediaPipe
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draw styled landmarks on the image
    """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    """
    Visualize prediction probabilities
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def send_pushover_notification(message):
    """
    Send notification via Pushover API
    """
    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": api_token,
        "user": user_key,
        "message": message,
    }
    requests.post(url, data=data)

def load_model():
    """
    Load the pre-trained model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    # Load weights from the file
    model.load_weights(MODEL_PATH)
    return model

def run_recognition():
    """
    Run the sign language recognition system
    """
    # Load model
    model = load_model()
    
    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    
    # Sequential action detection
    sequence_actions = []
    sequence_threshold = max(len(seq) for seq in target_sequences)
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action_index = np.argmax(res)
                predictions.append(action_index)
                
                # Sequential action detection logic
                sequence_actions.append(actions[action_index])
                sequence_actions = sequence_actions[-sequence_threshold:]
                
                # Check if any of the target sequences is detected
                for target_sequence in target_sequences:
                    if len(sequence_actions) >= len(target_sequence) and all(
                            action in sequence_actions[-len(target_sequence):] for action in target_sequence):
                        # Send Pushover notification
                        message = f"Camera action detected: {' '.join(target_sequence)} + call 911."
                        send_pushover_notification(message)
                        print(f"Notification sent: {message}")
                
                # Visualization logic
                if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == action_index:
                    if res[action_index] > threshold:
                        if len(sentence) > 0:
                            if actions[action_index] != sentence[-1]:
                                sentence.append(actions[action_index])
                        else:
                            sentence.append(actions[action_index])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                # Visualize probabilities
                image = prob_viz(res, actions, image, colors)
            
            # Display recognized signs
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('Sign Language Recognition', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
