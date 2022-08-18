import cv2
import os
import numpy as np
import paramset as P
import ColorFeature as CF
import CNNFeature as CnnFeature
import yolov5.detect_single as yolo_det


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

    if param[P.FEA_MODEL] == 'color':
        cv2.imwrite('output/{}_color_result.png'.format(filename), img)
    elif param[P.FEA_MODEL] == 'cnn':
        cv2.imwrite('output/{}_cnn_result.png'.format(filename), img)


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
                d = d + np.sum(np.abs(CF.get_block_feature(blocks[block_index]) - CF.get_block_feature(group_block)))
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
# 相邻的定义是: 在4连通区域相邻
def is_valid_groups(curGroup, group_diff_value, groups, param):
    if len(groups) < 1:
        return True

    # # group内部方差过大;
    # if np.array(group_diff_value).var() > 0.019:
    #     return False

    # 将已有的所有blocks index存放到temp
    BLOCKS_NUM = param[P.IMG_SUBBLOCK_NUM]
    temp = groups.copy()
    temp = list(np.array(temp).flatten())

    # 查看和已有的groups有多少相邻的
    c_neighbor = 0
    for block_index in curGroup:
        # 4连通区域
        neighbor_ind = [block_index-1, block_index+1, block_index-BLOCKS_NUM, block_index+BLOCKS_NUM]
        for i in range(4):
            if neighbor_ind[i] in temp:
                c_neighbor = c_neighbor + 1
                # 如果已有相邻的，则删除，以免后续还会与此block比对
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
    groups_diff_value = []
    group = []                   # 单独的一组block
    group_diff_value = []        # 单独的一组block之间的距离

    min_diffs = sorted(diffs)
    used_blocks = []             # 记录已经被选中的特征

    loop_num = 0
    slice_ratio = 0.9
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
                group_diff_value.append(diff.item())
            else:
                # 确认两组group中，是否有一个block是一样的，从而保证是相连区域;
                if len([i for i in group if i in block_index]) > 0:
                    group.extend(block_index)
                    group_diff_value.append(diff.item())
                    # 去掉重复的block编号
                    group = list(set(group))

            # 每一组x个block
            if len(group) == BLOCKS_PER_FEATURE:
                valid = is_valid_groups(group, group_diff_value, groups, param)
                if valid:
                    groups.append(group)
                    groups_diff_value.append(group_diff_value)
                    used_blocks.extend(group)
                    group = []
                    group_diff_value = []
                    loop_num = 0
                    break
                # 查找新一轮feature
                group = []
                group_diff_value = []
        loop_num += 1

        # 超过一定次数，还是没能组成指定数量block的一组
        # 则不再考虑这些blocks，直接放入已登记的组.同时降低
        # 特征组的要求
        if loop_num >= 6:
            used_blocks.extend(group)
            # 降低要求
            print("lower the requirement of connectivity!")
            param[P.FEA_BLOCKS_DIST] = param[P.FEA_BLOCKS_DIST]+1
            # print(used_blocks, groups)
            group = []
            group_diff_value = []
            loop_num = 0

    # groups信息
    print(groups)
    if param[P.DEBUG]:
        print(groups_diff_value)
        for g in groups_diff_value:
            print(np.array(g).var())

    # 选择两侧的blocks
    add_end_blocks(groups, blocks, border_block_index, used_blocks, param)

    return groups


# 查找宠物的头像区域
def find_pet_face(model, opt):
    res = yolo_det.run(model, **vars(opt))
    print(res)
    return res


def main_process(img, model, filename, param):
    # 将原图分为指定的子块;
    sub_block_num = param[P.IMG_SUBBLOCK_NUM]
    # N * N个blocks，每个blocks存储了该区域的子图像
    blocks = split_img(img, sub_block_num)

    # 获取每个子块的特征
    features = []
    features2 = []
    for b in blocks:
        # 颜色特征
        features.append(CF.get_block_feature(b))
        features2.append(CnnFeature.get_block_feature(b, model))

    # 计算每个子块之间的差异度
    diff_clr = CF.get_diff(img, features, param, filename)
    diff_cnn = CnnFeature.get_diff(img, features2, param, filename)

    # 根据差异度查找相邻的blocks,默认基于颜色查找
    diff = diff_clr
    if param[P.FEA_MODEL] == 'cnn':
        diff = diff_cnn
    groups = get_connected_blocks(diff, blocks, param)

    # 输出groups
    show_groups(img, groups, param, filename)
