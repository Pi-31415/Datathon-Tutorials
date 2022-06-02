# Datathon-Tutorials
Instance Segmentation Tutorials for Datathon

## Requirements

- [Detectron 2](https://github.com/facebookresearch/detectron2)
- [LabelMe](https://github.com/wkentaro/labelme/releases)
- [Intersection Over Union](https://medium.com/mlearning-ai/intersection-over-union-threshold-whats-the-purpose-of-using-it-and-how-it-helps-in-object-1a2d74de296f#:~:text=What%20is%20Intersection%20over%20Union,perfectly%20the%20image%20is%20segmented.)

## IOU

```python
from PIL import Image
from cv2 import IMREAD_REDUCED_GRAYSCALE_2
from numpy import asarray
import numpy as np

# load the image for Prediction
image = Image.open('pred.jpg')
# load the image for Target
image2 = Image.open('targ.jpg')

prediction = asarray(image)
target = asarray(image2)


intersection = np.logical_and(target, prediction)
union = np.logical_or(target, prediction)
iou_score = np.sum(intersection) / np.sum(union)

print(iou_score)

```


