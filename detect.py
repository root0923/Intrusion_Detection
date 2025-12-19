import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    '''
        和原始YOLOv11的detect相比，多了两个参数
            use_simotm="RGB",
            channels=3,
            
        Compared with the original "detect" in YOLOv11, it now has two additional parameters: 
            "use_simotm" set to "RGB". channels=3,
    '''
    model = YOLO(r"data\LLVIP-yolo11m-e300-16-pretrained.pt") # select your model.pt path
    model.predict(source=r'data\dataset\video_IR\test0.mp4',
                  imgsz=800,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=False,
                  use_simotm="RGB",
                  channels=3,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )