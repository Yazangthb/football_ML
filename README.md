# Football Analysis via ML

This project aims to perform advanced football video analysis using cutting-edge machine learning and computer vision techniques. The main objectives include detecting and tracking players, referees, and footballs in video footage, assigning players to teams based on their t-shirt colors, and analyzing player movements and performance metrics.


![Alt Text](inf_video.gif)

---

## Key Features
1. **Object Detection:** 
   - Detect and track players, referees, and footballs using YOLO, one of the leading AI object detection models.
   - Train YOLO for improved detection performance on football-related objects.

2. **Team Assignment:** 
   - Assign players to teams by analyzing t-shirt colors using KMeans clustering for pixel segmentation.
   - Calculate team ball possession percentages during matches.

3. **Camera Motion Estimation:** 
   - Use optical flow to measure camera movement between frames, enabling accurate player movement analysis.

4. **Perspective Transformation:** 
   - Apply perspective transformation to account for depth and perspective, allowing movement measurements in meters rather than pixels.

5. **Performance Metrics:** 
   - Calculate individual player speeds and distances covered during the game.

---

## Modules Used
This project employs the following modules:
- **YOLO:** AI object detection model.
- **KMeans:** Pixel segmentation and clustering to determine team t-shirt colors.
- **Optical Flow:** Analyze camera movement.
- **Perspective Transformation:** Measure scene depth and perspective.
- **Custom Utilities:** Compute player speed and distance.


## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas


