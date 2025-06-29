import os
import gzip
import subprocess
import json

def unzip_gz_files(gz_files):
    """
    解压 .gz 文件到指定目录或当前目录。
    :param gz_files: 包含 .gz 文件路径的列表或单个文件路径
    :param output_dir: 解压后的文件保存的目录（默认为当前目录）
    """
    if isinstance(gz_files, str):
        gz_files = [gz_files]
    for gz_file in gz_files:
        if not gz_file.endswith('.gz'):
            raise ValueError(f"文件 {gz_file} 不是 .gz 文件")
		# 获取.gz文件的当前所在目录
        output_dir = os.path.dirname(gz_file)
        # 提取文件名和目录路径
        file_name = os.path.basename(gz_file)[:-3]  # 去掉 .gz 并加上 .pickle
        output_file = os.path.join(output_dir, file_name)
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"文件 {gz_file} 已解压到 {output_file}")
        return output_file


# def run_genellm_inference(input_dict: dict, param_dict: dict, model_path: str,  output_path: str):
#     input_data = input_dict["input_path"]
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     # if not os.path.isfile(path):
#     #     output_path = os.path.join(output_path, "result.json")
#     #     os.makedirs(output_path)
#     if input_data.endswith('.gz'):
#         input_file = unzip_gz_files(input_data)
#     elif input_data.endswith('.pickle'):
#         input_file = input_data

#     command = ['bash', 'inference.sh', input_file, model_path, output_path]
#     # 执行命令
#     process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     # 获取命令执行的结果
#     stdout = process.stdout
#     stderr = process.stderr
#     # 打印输出和错误信息
#     print('Output:', stdout)
#     print('Error:', stderr)

#     # read results
#     with open(output_path, 'r', encoding='utf-8') as path:
#         data = json.load(path)

#     return "table", data


def run_genellm_inference(input_dict: dict, param_dict: dict, model_path: str,  output_path: str):
    input_data = input_dict["input_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "result.json")
    if input_data.endswith('.gz'):
        input_file = unzip_gz_files(input_data)
    elif input_data.endswith('.pickle'):
        input_file = input_data
    print("Start infering!")
    command = ['bash', 'inference.sh', input_file, model_path, output_file]
    # 执行命令
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # 获取命令执行的结果
    stdout = process.stdout
    stderr = process.stderr
    # 打印输出和错误信息
    print('Output:', stdout)
    print('Error:', stderr)

    # read results
    with open(output_file, 'r', encoding='utf-8') as path:
        data = json.load(path)
        
    return "table", data


if __name__ == "__main__":
    # input_data = "xxxx.pickle"               # 实际输入数据，文件
    # model_param_path = "xxxxx"      # 参数文件路径,文件夹，其中的参数文件名写死在代码中
    # mid_result_path = "xxxx"  # 中间结果文件路径，文件夹
    

    input_data = {"input_path":"../examples/P0240428101002_1_1.pickle.gz"}
    model_param_path = "../parameters"
    mid_result_path = "../results"
    param_dict = None
    run_genellm_inference(input_data, param_dict, model_param_path, mid_result_path)