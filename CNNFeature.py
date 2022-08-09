import cv2
import PIL.Image as Image
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
import paramset as P


# 获得每个子块的特征
def get_block_feature(block, model):
    img = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
    feature = model(img)

    return feature


# 计算并返回归一化后的子块之间的diff。
# diff和相似度不同，diff越大,相似度越小。cnn计算的是相似度，colorfeature计算的是diff
def get_diff(img, features, param, filename):
    BLOCK_NUM = param[P.IMG_SUBBLOCK_NUM]

    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[..., 3] = 100
    piece_width = int(img.shape[0] / BLOCK_NUM)
    piece_high = int(img.shape[1] / BLOCK_NUM)

    # 计算每个block特征之间的极值差异，为后续标准化准备
    similarity = []
    for i in range(0, len(features) - 1):
        # 左右
        sim = F.cosine_similarity(features[i], features[i+1])
        similarity.append(sim)
        # 上下
        if i + BLOCK_NUM < len(features):
            sim = F.cosine_similarity(features[i], features[i + BLOCK_NUM])
            similarity.append(sim)
    max_sim = max(similarity)
    min_sim = min(similarity)

    sorted_diff = []
    line_width = 20
    # 计算与右方块的diff
    for i in range(0, BLOCK_NUM):
        for j in range(0, BLOCK_NUM - 1):
            # diff = np.sum(np.abs(features[i * BLOCK_NUM + j] - features[i * BLOCK_NUM + j + 1]))
            sim = F.cosine_similarity(features[i * BLOCK_NUM + j], features[i * BLOCK_NUM + j + 1])
            # 归一化
            ratio = (sim - min_sim) / (max_sim - min_sim)
            # 这里余弦距离越小，表示两者相似性越小，差距diff越大
            sorted_diff.append(1-ratio)

            # 利用透明度来表示相似性，越亮相似度越大
            img[i * piece_width:(i + 1) * piece_width,
            (j + 1) * piece_high - int(line_width / 2):
            (j + 1) * piece_high + int(line_width / 2)
            ] = [255 * ratio, 255 * ratio, 255 * ratio, 255]

    # 计算与下方块的diff
    for i in range(0, BLOCK_NUM - 1):
        for j in range(0, BLOCK_NUM):
            # diff = np.sum(np.abs(features[i * BLOCK_NUM + j] - features[(i + 1) * BLOCK_NUM + j]))
            sim = F.cosine_similarity(features[i * BLOCK_NUM + j], features[(i+1) * BLOCK_NUM + j])
            ratio = (sim - min_sim) / (max_sim - min_sim)
            sorted_diff.append(1-ratio)

            img[(i + 1) * piece_width - int(line_width / 2):
                (i + 1) * piece_width + int(line_width / 2),
            j * piece_high:(j + 1) * piece_high
            ] = [255 * ratio, 255 * ratio, 255 * ratio, 255]

    if param[P.DEBUG]:
        cv2.imwrite('output/{}_cnn_sim.png'.format(filename), img)

    return sorted_diff

