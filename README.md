# Detection
Read here: https://github.com/kbardool/keras-frcnn
- In that path must contains 3 directories (csv/test/train)
- In csv directory contains test/train/ann.csv, train.txt and test/train csv header is ```filename, weight, food```,  ann csv heaader is ```image, xmin, ymin, xmax, ymax, food```
- In train/test directories contain cropped image of each food

# Estimation
```python calorie_estimation.py --path path --option train/test```
- In that path must contains 3 directories (csv/test/train)
- In csv directory contains test/train.csv and csv header is ```filename, weight, food```
- In train/test directories contain image of divided food tray
# Note
