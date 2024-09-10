# Human Detection Model

## Overview

This project includes a Python-based human detection model that utilizes the MobileNet SSD model for detecting humans in real-time video streams. It processes video from a connected webcam or camera, detects human figures, and highlights them with bounding boxes.

## Features

- Real-time human detection
- Bounding box around detected humans
- Audio alert (beep sound) when a human is detected
- Frame skipping and resizing for optimized performance

## Requirements

- Python 3.x
- OpenCV
- Numpy

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/japneetsingh035/HumanDetectionModel.git
   ```

2. **Install Dependencies**

   Make sure you have Python installed. Then, install the required libraries using pip:

   ```bash
   pip install opencv-python numpy
   ```

## Setup

1. **Download Pre-trained Model Files**

   You need the MobileNet SSD model and config files. Download them from the following links:

   Place these files in the `path/to/` directory or adjust the paths in the script accordingly.

2. **Update Script Paths**

   Ensure the paths to the model and config files in the script are correctly set:

   ```python
   model = "path/to/MobileNetSSD_deploy.caffemodel"
   config = "path/to/MobileNetSSD_deploy.prototxt"
   ```

## Usage

1. **Run the Script**

   Execute the script to start human detection:

   ```bash
   python human_detection.py
   ```

2. **Viewing Output**

   The video feed will display with bounding boxes drawn around detected humans. The script will beep when a human is detected.

3. **Stopping the Script**

   Press 'q' while the video window is active to stop the script.

## Troubleshooting

- **No Frame Captured**: Ensure your webcam or camera is properly connected and accessible. Adjust the `VideoCapture` index if needed.
- **Model Loading Issues**: Verify that the model and config file paths are correct and that the files are not corrupted.

## Contributing

Feel free to fork the repository and submit pull requests. For issues or feature requests, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
