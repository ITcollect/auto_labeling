# Auto Labeling Tool

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/github/license/ITcollect/auto_labeling)

自动化数据标注工具，支持快速生成图像/文本数据的标签。

## 功能特性

- 🖼️ **图像标注**：支持矩形框、多边形标注
- 📝 **文本标注**：实体识别、分类标签
- ⚡ **自动化辅助**：基于模型的预标注功能
- 🔄 **格式转换**：支持 COCO、YOLO、Pascal VOC 等格式导出

## 安装指南

### 依赖环境
- Python 3.7+
- OpenCV
- PyQt5

### 安装步骤
```bash
git clone https://github.com/ITcollect/auto_labeling.git
cd auto_labeling
pip install -r requirements.txt
