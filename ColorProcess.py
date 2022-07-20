import cv2
import os
import numpy as np
import paramset as P


COLORS = [
    [243, 178, 207, 255],
    [102, 255, 108, 255],
    [0, 232, 232, 255],
    [207, 234, 0, 255],
    [67, 24, 155, 255],
    [55, 97, 123, 255],
    [28, 97, 255, 255],
    [108, 211, 34, 255],
]


def get_block_info(param):
    BLOCKS_NUM = param['sub_block_num']

    block_map= [-1] * (BLOCKS_NUM - 1) * BLOCKS_NUM * 2
    border_block_index = []

    # 边界block的index
    for i in range(0, BLOCKS_NUM):
        # 上面一行
        border_block_index.append(i)
        # 左边一列
        border_block_index.append(i * BLOCKS_NUM)
        # 右边一列
        border_block_index.append(i * BLOCKS_NUM - 1)
        # 下面一行
        border_block_index.append(BLOCKS_NUM * BLOCKS_NUM - 1 - i)

    # 将blcok之间的水平diff记录在blockmap中
    # blockmap[i]=[左边block的index, 右边block的index]
    for i in range(0, BLOCKS_NUM):
        for j in range(0, BLOCKS_NUM - 1):
            block_map[i * (BLOCKS_NUM - 1) + j] = [i * BLOCKS_NUM + j, i * BLOCKS_NUM + j + 1]

    # 延续上述blockmap表，将block之间的竖直diff记录在blockmap中
    # blockmap[i]=[上边block的index, 下边block的index]
    for i in range(0, BLOCKS_NUM - 1):
        for j in range(0, BLOCKS_NUM):
            block_map[BLOCKS_NUM * (BLOCKS_NUM - 1) + i * BLOCKS_NUM + j] = [i * BLOCKS_NUM + j,
                                                                             (i + 1) * BLOCKS_NUM + j]

    return block_map, border_block_index


def get_Several_MinMax_Array(np_arr, several):
    if several > 0:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max


# 将原图分为num*num个子块
def split_img(img, sub_block_num):
    piece_width = int(img.shape[0] / sub_block_num)
    piece_high = int(img.shape[1] / sub_block_num)
    pieces = []
    for i in range(0, sub_block_num):
        for j in range(0, sub_block_num):
            pieces.append(
                img[
                i * piece_width:(i + 1) * piece_width,
                j * piece_high:(j + 1) * piece_high,
                ]
            )
    return pieces


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
        diff = np.sum(np.abs(features[i] - features[i + 1]))
        diffs.append(diff)
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
            ] = [255 * (1 - ratio), 255 * (1 - ratio), 255, 255]

    # 计算与下方块的diff
    for i in range(0, BLOCK_NUM - 1):
        for j in range(0, BLOCK_NUM):
            diff = np.sum(np.abs(features[i * BLOCK_NUM + j] - features[(i + 1) * BLOCK_NUM + j]))
            ratio = (diff - min_diff) / (max_diff - min_diff)
            sorted_diffs.append(ratio)

            img[(i + 1) * piece_width - int(line_width / 2):
                (i + 1) * piece_width + int(line_width / 2),
            j * piece_high:(j + 1) * piece_high
            ] = [255 * (1 - ratio), 255 * (1 - ratio), 255, 255]

    # cv2.imwrite('output/{}_diff.png'.format(filename), img)
    return sorted_diffs


def show_groups(img, groups, param, filename):
    BLOCKS_NUM = param[P.IMG_SUBBLOCK_NUM]

    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[..., 3] = 255
    origin = img.copy()

    piece_width = int(img.shape[0] / BLOCKS_NUM)
    piece_high = int(img.shape[1] / BLOCKS_NUM)
    line_width = 20
    for index, group in enumerate(groups):
        color = COLORS[index]
        for block_index in group:
            i = int(block_index / BLOCKS_NUM)
            j = block_index % BLOCKS_NUM
            img[i * piece_width:(i + 1) * piece_width, j * piece_width:(j + 1) * piece_width] = color
            img[i * piece_width + line_width:(i + 1) * piece_width - line_width,
            j * piece_width + line_width:(j + 1) * piece_width - line_width] \
                = origin[i * piece_width + line_width:(i + 1) * piece_width - line_width,
                  j * piece_width + line_width:(j + 1) * piece_width - line_width]

    img[0:4, 0:img.shape[1]] = [0, 0, 0, 255]
    img[0:img.shape[0], 0:4] = [0, 0, 0, 255]
    for i in range(0, BLOCKS_NUM + 1):
        img[i * piece_width - 4:i * piece_width + 4, 0:img.shape[1]] = [0, 0, 0, 255]
        img[0:img.shape[0], i * piece_width - 4:i * piece_width + 4, ] = [0, 0, 0, 255]

    if not os.path.exists("output"):
        os.makedirs("output")
    cv2.imwrite('output/{}_result.png'.format(filename), img)


# 为每个group添加两端的block
def add_end_blocks(groups, blocks, border_block_index, used_blocks, param):
    # 部分参数
    BLOCKS_NUM = param[P.IMG_SUBBLOCK_NUM]

    all_index = range(0, BLOCKS_NUM * BLOCKS_NUM)

    # 为每一组找左右两个block
    for group in groups:
        available_indexs = []
        for block_index in all_index:
            # 边界block
            if block_index in border_block_index:
                continue
            # 已登记的block
            if block_index in used_blocks:
                continue

            # 与当前特征中的block相邻的block也不可以
            neighbor_block = False
            for i in range(len(group)):
                if abs(block_index / BLOCKS_NUM - group[i] / BLOCKS_NUM) + abs(
                        block_index % BLOCKS_NUM - group[i] % BLOCKS_NUM) < 2:
                    neighbor_block = True
                    break
            if neighbor_block:
                continue
            available_indexs.append(block_index)

        # 和每组特征之间的总的距离
        dist = []
        for block_index in available_indexs:
            d = 0
            for i in range(len(group)):
                group_block = blocks[group[i]]
                d = d + np.sum(np.abs(get_block_feature(blocks[block_index]) - get_block_feature(group_block)))
            dist.append(d)

        # 将与group距离最大的blcok作为左右两端的block
        for i in range(2):
            ind = dist.index(max(dist))
            block_ind = available_indexs[ind]
            group.append(block_ind)
            used_blocks.append(block_ind)

            dist.pop(ind)
            available_indexs.pop(ind)

    return groups


# 确认当前的group是否符合;主要检查是否是相邻的
def is_valid_groups(curGroup, groups, param):
    if len(groups) < 1:
        return True

    BLOCKS_NUM = param[P.IMG_SUBBLOCK_NUM]
    temp = groups.copy()
    temp = list(np.array(temp).flatten())

    # 查看和已有的groups有多少相邻的
    c_neighbor = 0
    for block_index in curGroup:
        neighbor_ind = [block_index-1, block_index+1, block_index-BLOCKS_NUM, block_index+BLOCKS_NUM]
        for i in range(4):
            if neighbor_ind[i] in temp:
                c_neighbor = c_neighbor + 1
                temp.remove(neighbor_ind[i])

    # 与已有的groups最多只能有多少个相邻的blocks
    dist = param[P.FEA_BLOCKS_DIST]
    return c_neighbor <= dist


# 根据相似度查找比较接近且相连的子块
def get_connected_blocks(diffs, blocks, param):
    # 部分参数
    GROUP_NUM = param[P.IMG_FEATURES_NUM]
    BLOCKS_PER_FEATURE = param[P.FEA_BLOCKS_NUM]

    groups = []                  #存储每组选好的特征
    min_diffs = sorted(diffs)
    used_blocks = []             # 记录已经被选中的特征
    group = []                   # 单独的一组特征
    loop_num = 0
    slice_ratio = 0.8
    block_map, border_block_index = get_block_info(param)
    while len(groups) < GROUP_NUM:
        for diff in min_diffs[:int(len(min_diffs) * slice_ratio)]:
            # 获得当前diff对应的两个block的index
            block_index = block_map[diffs.index(diff)]
            # 边界不考虑
            if len([i for i in block_index if i in border_block_index]) > 0:
                continue
            # 已经登记了的不考虑
            if len([i for i in block_index if i in used_blocks]) > 0:
                continue
            # 初次登记
            if len(group) == 0:
                group.extend(block_index)
            else:
                # 确认两组group中，是否有一个block是一样的，从而保证是相连区域;
                if len([i for i in group if i in block_index]) > 0:
                    group.extend(block_index)
                    # 去掉重复的block编号
                    group = list(set(group))

            # 每一组x个block
            if len(group) == BLOCKS_PER_FEATURE:
                valid = is_valid_groups(group, groups, param)
                if valid:
                    groups.append(group)
                    used_blocks.extend(group)
                    loop_num = 0
                    break
                group = []
        loop_num += 1

        # 超过一定次数，还是没能组成指定数量block的一组
        # 则不再考虑这些blocks，直接放入已登记的组
        if loop_num >= 5:
            used_blocks.extend(group)
            group = []
            loop_num = 0
    print(groups)

    # 选择两侧的blocks
    add_end_blocks(groups, blocks, border_block_index, used_blocks, param)

    return groups


def main_process(img, filename, param):
    # 将原图分为指定的子块;
    sub_block_num = param[P.IMG_SUBBLOCK_NUM]
    blocks = split_img(img, sub_block_num)

    # 获取每个子块的特征
    features = []
    for b in blocks:
        features.append(get_block_feature(b))

    # 计算每个子块之间的相似度
    similarity = get_diff(img, features, param, filename)

    # 根据相似度查找相邻的blocks
    groups = get_connected_blocks(similarity, blocks, param)

    # 输出groups
    show_groups(img, groups, param, filename)
