## 人脸表情识别问题

---

#### 文件结构

+ `data/`：文件夹下是训练集，以及一些图表、评估结果、网络应用所需的图片、视频文件
+ `trained/`：文件夹下是训练好的模型（带网络结构），以及opencv识别人脸位置的模型
+ `Model.py`：网络结构
+ `DataLoader.py`：从csv文件中导入数据
+ `Eval.py`：对于训练好的模型进行评估
+ `Application.py`：对于训练好的模型应用到图片和视频中
+ `expression_recognition.ipynb`：训练评估全过程，最下方有评估结果
+ `requirements.txt`：依赖环境

---

#### `Application.py`的使用

+ 环境要求

见[requirements.txt](./requirements.txt)

+ 获取帮助

```shell
python Application.py -h
```

+ usage

```shell
python Application.py --path [FILE_PATH]
```

+ example

```sh
python Application.py --path ./data/img.jpg
```

```sh
python Application.py --path ./data/video.mp4
```

