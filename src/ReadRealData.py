# import pandas as pd
# import os
# import glob
# import pickle

# class PatientDataLoader:
#     def __init__(self,input_data):
#         """
#         初始化加载器
#         :param excel_path: Excel文件的路径
#         :param sheet_name: 子表名称列表
#         """
#         # self.excel_path = excel_path
#         # self.sheet_list = sheet_list
#         self.input_data = input_data
        

#     def get_pretrain_list(self, base_dir):
#         """
#         加载每个患者的数据
#         :param base_dir: 存放患者数据的基础目录
#         :return: 生成器，逐个返回患者数据
#         """
#         pretrain_list = []
#         # for sheet_name in self.sheet_list:
#         #     #print("tackling",sheet_name,flush=True)
#         #     df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
#         #     for index, row in df.iterrows():
#         #         patient_id = str(row[0])
#         #         date = str(row[1])
#                 date_path = self.input_data
#                 # folder_path = os.path.join(base_dir, date_path)
#                 # files = os.listdir(folder_path)
#                 # for file in files:
#                     # if file.startswith(patient_id):
#                         file_path = os.path.join(folder_path,file)
#                         pretrain_list.append(file_path)
#                         #print(file_path,flush=True)
#                         # the pickle file is a list contains several(around 36000000 - 56000000)
#                         # DNA sequences(length around 100~200)
#                         #with open(file_path, 'rb') as file:
#                         #    data = pickle.load(file)
#                         #    print(type(data),flush=True)
#                         #    print(len(data),flush=True)
#                         #    print(data[0],flush=True)
#         return pretrain_list

# if __name__ == '__main__':
#     focus_6G_list = ['健康6G','结直肠癌6G','胃癌6G','肺癌6G','肝癌6G']
#     base_dir = '../data_provider/OxTium_data'
#     loader = PatientDataLoader('qc统计20240125.xlsx', focus_6G_list)
#     a = loader.get_pretrain_list(base_dir)
