# Real-Time Object Detection Model using Machine Learning

## Overview

This project focuses on developing an efficient and accurate system capable of detecting various objects in real-time video streams and extracting number plates from vehicles. By integrating computer vision algorithms with machine learning techniques, the system addresses challenges in real-time object detection and number plate recognition in practical scenarios.

## Features

- **Real-Time Object Detection**: Utilizes pre-trained YOLOv4 models to detect objects in live video streams.
- **Number Plate Recognition**: Extracts and recognizes vehicle number plates from detected objects.
- **Custom Object Counting**: Counts total objects detected or the number of objects per class.
- **Detailed Detection Information**: Provides class, confidence, and bounding box coordinates for each detection.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (for environment management)
- [Python 3.x](https://www.python.org/downloads/)

### Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Paarth01/Real-Time-Object-Detection-Model-using-Machine-Learning-.git
    cd Real-Time-Object-Detection-Model-using-Machine-Learning-
    ```

2. **Set Up the Environment**:

    - For TensorFlow CPU:

        ```bash
        conda env create -f conda-cpu.yml
        conda activate yolov4-cpu
        ```

    - For TensorFlow GPU:

        ```bash
        conda env create -f conda-gpu.yml
        conda activate yolov4-gpu
        ```

    Alternatively, using `pip`:

    - For TensorFlow CPU:

        ```bash
        pip install -r requirements.txt
        ```

    - For TensorFlow GPU:

        ```bash
        pip install -r requirements-gpu.txt
        ```

3. **NVIDIA Driver (For GPU Users)**:

    Ensure CUDA Toolkit version 10.1 is installed, as it is compatible with the TensorFlow version used in this project.

### Download Pre-trained Weights

The YOLOv4 model comes pre-trained and can detect 80 classes. For demonstration purposes, use the pre-trained weights.

### Running the Model

Execute the following commands to run the object detection model:

```bash
python yolo_project.py --weights_location ./weights/yolov4-tiny-416 --model yolov4 --video_location cars_test.mp4
