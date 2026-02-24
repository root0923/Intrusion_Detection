import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/ppyoloe/ppyoloe-s.yaml')
    # # model.load('yolov8n.pt') # loading pretrain weights
    model = YOLO('data/LLVIP-yolo11m-e300-16-pretrained.pt')
    model.train(data=R'ultralytics/cfg/datasets/lake.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                close_mosaic=5,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='BCCD',
                name='BCCD-ppyoloe-s',
                )