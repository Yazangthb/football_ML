from ultralytics import YOLO
model = YOLO('models/best.pt')
results = model.predict('input_videos/A1606b0e6_0 (12).mp4', save=True)
print(results[0])
print('==================================')
for box in results[0].boxes:
    print(box)