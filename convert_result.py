import pandas as pd

path = "../deepSort_result/Text.txt"
# deepsort_result = "../deepSort_result/result.txt"
# no_small_bb_result = "../deepSort_result/deepsort_result_removed_smallbb.txt"
# yolov5_result = "../deepSort_result/yolov5_deepsort.txt"

df = pd.read_csv(path, sep=" +", index_col=0)
# df1 = pd.read_csv(deepsort_result, sep=" +", index_col=0)
# df2 = pd.read_csv(no_small_bb_result, sep=" +", index_col=0)
# df3 = pd.read_csv(yolov5_result, sep=" +", index_col=0)

# print(df.columns)
# print(df1.head())

df.to_excel('deepSORT_metrics.xlsx')
# df1.to_excel('fastRCNN.xlsx')
# df2.to_excel('removed_smallBB_maskRCNN.xlsx')
# df3.to_excel('yolov5_deepsort.xlsx')