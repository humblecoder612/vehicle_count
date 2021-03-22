# Vehicle Detector
## Counting Vehicles Passing line


I have used EfficientDet which is one of the state of the art object detector in whihc vehicles are detected by pretrained coco weights. I have choose this because of using its D1 version, it gives good accuracy with good Fps that means less computational power.

EfficientDets are a family of object detection models, which achieve state-of-the-art 55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors.


<img src="https://github.com/google/automl/blob/master/efficientdet/g3doc/flops.png" width="800" />

I was trying to use Yolo V5 but the output size was turning out big , so I decided with this model as results were similar almost.


## Technique
> The bounding box of the vehicle which is detected , it's centroid is used for the vehicle count if centroid passes throught the line , the count will increase.

Here is the output videos :

https://drive.google.com/drive/folders/1OjMhOasga9MTCvZ_5gnGW81TMZ7xflwy?usp=sharing



## Steps
Go to terminal :
- Git clone - https://github.com/xuannianz/EfficientDet
- Install the required libraries
- run the vehichlecount.py in terminal like : python vehicle_count.py --source filepath --x1 x1 --y1 y1 --x2 x2 --y2 y2

## Outputs

<img src="https://github.com/humblecoder612/vehicle_count/blob/main/Screenshot_20210322_115923.png" width="800" />

