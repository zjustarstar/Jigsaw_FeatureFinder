import os
import cv2
from tqdm import tqdm
import paramset as P
import MainProcess as mp


def main():
    dir_name = 'testimg'

    # 参数
    param = {}
    # 调试模式;会生成一些中间数据
    param[P.DEBUG] = False
    # 特征模式
    param[P.FEA_MODEL] = 'color'
    # 每行每列分为多少个子块
    param[P.IMG_SUBBLOCK_NUM] = 8
    # 生成多少组特征
    param[P.IMG_FEATURES_NUM] = 3
    # 每一组多少个子块，默认每4个组成一个feature
    param[P.FEA_BLOCKS_NUM] = 4
    # 每两个特征之间的距离
    param[P.FEA_BLOCKS_DIST] = 0

    # 加载model
    model = mp.load_cnn_model()

    for name, l, fl in os.walk(dir_name):
        for filename in tqdm(fl):
            print("当前正在处理:{}".format(filename))
            img = cv2.imread('{}/{}'.format(dir_name, filename), -1)
            filename = filename[:-4]
            mp.main_process(img, model, filename, param)


if __name__ == '__main__':
    main()
