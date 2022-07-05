import os
import cv2
from tqdm import tqdm
import ColorProcess as cp


def main():
    dir_name = 'testimg'

    # 参数
    param = {}
    # 每行每列分为多少个子块
    param['sub_block_num'] = 8
    # 生成多少组特征
    param['features_num'] = 3
    # 每一组多少个子块，默认每4个组成一个feature
    param['blocks_per_feature'] = 4

    for name, l, fl in os.walk(dir_name):
        for filename in tqdm(fl):
            print("当前正在处理:{}".format(filename))
            img = cv2.imread('{}/{}'.format(dir_name, filename), -1)
            filename = filename[:-4]
            cp.main_process(img, filename, param)


if __name__ == '__main__':
    main()
