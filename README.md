# ADSB-Injection-Detection
An Artificial Intelligence Solution for Detecting and Classifying Automatic Dependent Surveillance-broadcast Message Injection Cyberattacks: A Study Targeting Major U.S. Commercial Airlines and International Airports

### Attack Visualization
This folder contains video examples of simulated injection attacks. The project to visualize the attacks can be found [here](https://github.com/redwards24/ADS-B-Data-Collector).

### Dataset Collection and Attack Generation
This folder contains the necessary files to collect and filter authentic state vectors from OpenSky as well as generate three types of attacks:
1. Path Modification (PM)
2. Ghost Injection (GI)
3. Velocity Drift(VD)

### Deep Learning
This folder contains the code for training and evaluating several deep learning models for attack classification. The models used for this portion were:
1. TabNet
2. Simple 1D CNN
3. ResNet50
4. EfficientNetB0
5. VGG-16
6. AlexNet

### Machine Learning
This folder contains the code for training and evaluating six machine learning models for attack classification.
The models tested were:
1. Random Forest
2. Decision Tree
3. Naive Bayes
4. Multi-layer Perceptron
5. Logistic Regression
6. K-Nearest Neighbors

### Reciever Implementation
This folder contains a video tutorial on how to set up a Raspberry-Pi with an ADS-b reciever to collect and upload real-time ADS-b messages to OpenSky. [PiAware](https://www.flightaware.com/adsb/piaware/build) was used for this process.
