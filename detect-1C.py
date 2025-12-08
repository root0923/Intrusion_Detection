import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    '''
        和原始YOLOv11的detect相比，多了两个参数
            use_simotm="Gray",
            channels=1,

        Compared with the original "detect" in YOLOv11, it now has two additional parameters: 
            "use_simotm" set to "Gray". channels=1,
    '''

    model = YOLO(R"data/LLVIP_IF-yolo11x-e300-16-pretrained.pt") # select your model.pt path
    model.predict(source=r'data\dataset\infrared',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGB", # Gray: uint8  Gray16bit: uint16
                  channels=3,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )