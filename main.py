import os
import cv2
from pathlib import Path
from tqdm import tqdm
import paramset as P
import yolov5.detect_single as yolo_det
import CNNFeature as CNN
import MainProcess as mp


def main():
    dir_name = 'testimg'

    # 参数
    param = {}
    # 调试模式;会生成一些中间数据
    param[P.DEBUG] = False
    # 特征模式
    param[P.FEA_MODEL] = 'cnn'
    # 每行每列分为多少个子块
    param[P.IMG_SUBBLOCK_NUM] = 8
    # 生成多少组特征
    param[P.IMG_FEATURES_NUM] = 2
    # 每一组多少个子块，默认每4个组成一个feature
    param[P.FEA_BLOCKS_NUM] = 6
    # 每两个特征之间的距离
    param[P.FEA_BLOCKS_DIST] = 0

    # 加载特征检测model
    model_feature = CNN.load_cnn_model()

    # 加载yolo model
    opt = yolo_det.parse_opt()

    # yolo目标检测
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    opt.nosave = True   # 不保存图像
    opt.exist_ok = True # 不新增保存结果的路径
    opt.data = ROOT / 'yolov5/models/pet.yaml'
    opt.weights = ROOT / "yolov5/weights/pet_face.pt"
    pet_face_model = yolo_det.load_model(**vars(opt))

    for name, l, fl in os.walk(dir_name):
        for filename in tqdm(fl):
            print("当前正在处理:{}".format(filename))
            img = cv2.imread('{}/{}'.format(dir_name, filename), -1)

            pathname = os.path.join(os.path.join(ROOT, dir_name), filename)
            opt.source = pathname
            pet_box = mp.find_pet_face(pet_face_model, opt, img, param)

            # 去掉末尾的.jpg
            filename = filename[:-4]
            mp.main_process(img, model_feature, filename, param, pet_box)


if __name__ == '__main__':
    main()
