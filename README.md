# Drowsiness Alert System

A real-time drowsiness detection system that uses facial landmarks to monitor the driver's eye activity. The system detects whether the driver is drowsy based on Eye Aspect Ratio (EAR) and provides an alert using audio and visual warnings when the driver shows signs of drowsiness. The project leverages computer vision techniques using OpenCV and Dlib for face and eye detection.

## Features
- Real-time eye tracking to determine drowsiness based on Eye Aspect Ratio (EAR).
- Displays "DROWSY!" on the screen with a red background when drowsiness is detected.
- Plays an audio alert when drowsiness is detected.
- Displays "AWAKE" with a green background when the eyes are open and awake.
- Audio alert stops when the driver is awake.
  
## Requirements
- Python 3.x
- OpenCV
- Dlib
- Numpy
- Scipy
- Playsound
- Shape predictor file for facial landmarks: `shape_predictor_68_face_landmarks.dat`

### Install the necessary libraries:
To install the required Python libraries, run the following command:

```bash
pip install opencv-python dlib numpy scipy playsound
```
### Download Shape Predictor:
You can download the shape_predictor_68_face_landmarks.dat file from the following link: https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat

Shape Predictor 68 Face Landmarks
Unzip the file and place it in the project directory.

### How to Run
Ensure that the necessary Python libraries are installed.
Download the shape_predictor_68_face_landmarks.dat file and place it in the project directory.
Place the alert sound file (alert_sound.mp3) in the assets folder.
Run the main.py script to start the drowsiness detection system.
```bash
python main.py
```
The system will start the webcam feed. When the eyes are closed for a certain period, a warning message "DROWSY!" will be displayed with an audio alert. When the eyes are open, the message "AWAKE" will be displayed.

### Project Structure
```
Drowsiness_Alert_System/
│
├── assets/
│   └── alert_sound.mp3
├── shape_predictor_68_face_landmarks.dat
├── main.py
├── README.md
```
### Future Enhancements
- Improved user interface for displaying warnings.
- Integration with a car's onboard system for real-time driver monitoring.

### Acknowledgements
- Dlib for face and landmark detection.
- OpenCV for real-time computer vision processing.
- Scipy for distance calculation.

### Conclusion
The Drowsiness Alert System is an effective solution for detecting driver drowsiness using computer vision techniques. It helps in enhancing road safety by providing timely alerts to drivers. This project serves as a foundation for building more advanced driver monitoring systems by integrating additional machine learning models or using more sophisticated sensors.

