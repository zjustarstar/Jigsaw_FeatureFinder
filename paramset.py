# 一些变量

# 关于图像的一些变量
IMG_SUBBLOCK_NUM = 'sub_block_num'     # 每行每列分为多少个子块
IMG_FEATURES_NUM = 'features_num'      # 关于特征的一些变量

# 关于特征的一些变量
FEA_BLOCKS_NUM = 'blocks_per_feature'  # 每一组多少个子块，默认每4个组成一个feature
FEA_BLOCKS_DIST = 'feature_distance'   # 每个特征之间的距离。两个block处于四连通区域距离为1
FEA_MODEL = 'feature_model'            # 特征模式:color, cnn, both

# 其它
DEBUG = False   # 用于调试