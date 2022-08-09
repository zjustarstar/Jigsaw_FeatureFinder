import cv2
import numpy as np
import paramset as P


# 获得每个子块的特征
def get_block_feature(block, index=-1):
    # 将每个子块分为3*3区域，分别计算其颜色特征
    block_width = int(block.shape[0] / 3)
    block_high = int(block.shape[1] / 3)

    result = []
    for i in range(0, 3):
        for j in range(0, 3):
            result.append([
                np.average(block[i * block_width:(i + 1) * block_width, j * block_high:(j + 1) * block_high, 0]),
                np.average(block[i * block_width:(i + 1) * block_width, j * block_high:(j + 1) * block_high, 1]),
                np.average(block[i * block_width:(i + 1) * block_width, j * block_high:(j + 1) * block_high, 2])
            ])
    result = np.array(result)

    if index == -1:
        return result.reshape(result.shape[0] * result.shape[1])

    blur_piece = np.zeros(block.shape)

    for i in range(0, 3):
        for j in range(0, 3):
            blur_piece[
            i * block_width:(i + 1) * block_width, j * block_high:(j + 1) * block_high
            ] = result[i * 3 + j]

    cv2.imwrite('output/blur_piece_{}.png'.format(index), blur_piece)
    return result.reshape(result.shape[0] * result.shape[1])


# 计算并返回归一化后的子块之间的diff
def get_diff(img, features, param, filename):
    BLOCK_NUM = param[P.IMG_SUBBLOCK_NUM]

    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[..., 3] = 100
    piece_width = int(img.shape[0] / BLOCK_NUM)
    piece_high = int(img.shape[1] / BLOCK_NUM)

    # 计算每个block特征之间的极值差异，为后续标准化准备
    diffs = []
    for i in range(0, len(features) - 1):
        # 左右
        diff = np.sum(np.abs(features[i] - features[i + 1]))
        diffs.append(diff)
        # 上下
        if i + BLOCK_NUM < len(features):
            diff = np.sum(np.abs(features[i] - features[i + BLOCK_NUM]))
            diffs.append(diff)
    max_diff = max(diffs)
    min_diff = min(diffs)

    sorted_diffs = []
    line_width = 20
    # 计算与右方块的diff
    for i in range(0, BLOCK_NUM):
        for j in range(0, BLOCK_NUM - 1):
            diff = np.sum(np.abs(features[i * BLOCK_NUM + j] - features[i * BLOCK_NUM + j + 1]))
            # 归一化
            ratio = (diff - min_diff) / (max_diff - min_diff)
            sorted_diffs.append(ratio)

            # 利用透明度来表示相似性
            img[i * piece_width:(i + 1) * piece_width,
            (j + 1) * piece_high - int(line_width / 2):
            (j + 1) * piece_high + int(line_width / 2)
            ] = [255 * (1 - ratio), 255 * (1 - ratio), 255 * (1 - ratio), 255]

    # 计算与下方块的diff
    for i in range(0, BLOCK_NUM - 1):
        for j in range(0, BLOCK_NUM):
            diff = np.sum(np.abs(features[i * BLOCK_NUM + j] - features[(i + 1) * BLOCK_NUM + j]))
            ratio = (diff - min_diff) / (max_diff - min_diff)
            sorted_diffs.append(ratio)

            img[(i + 1) * piece_width - int(line_width / 2):
                (i + 1) * piece_width + int(line_width / 2),
            j * piece_high:(j + 1) * piece_high
            ] = [255 * (1 - ratio), 255 * (1 - ratio), 255 * (1 - ratio), 255]

    if param[P.DEBUG]:
        cv2.imwrite('output/{}_cf_diff.png'.format(filename), img)

    return sorted_diffs

