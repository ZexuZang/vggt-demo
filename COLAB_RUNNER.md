# Colab Runner for VGGT

在 Colab 里按顺序运行下面这些 cell。

## 1. 检查 GPU

```python
!nvidia-smi
```

## 2. 克隆你的 VGGT 仓库

第一次运行：

```python
!git clone https://github.com/ZexuZang/vggt.git
%cd vggt
```

如果已经 clone 过：

```python
%cd /content/vggt
!git pull
```

## 3. 安装依赖

```python
!pip install -r requirements.txt
!pip install -r requirements_demo.txt
```

## 4. 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 5. 运行你在 VS Code 写好的主脚本

请把下面的路径改成你自己的 Drive 路径。

```python
!python scripts/run_vggt_colmap.py \
  --source_scene_dir "/content/drive/MyDrive/vggt_data/my_scene" \
  --local_scene_dir "/content/my_scene" \
  --output_scene_dir "/content/drive/MyDrive/vggt_outputs/my_scene" \
  --clean_local
```

如果想开启 bundle adjustment：

```python
!python scripts/run_vggt_colmap.py \
  --source_scene_dir "/content/drive/MyDrive/vggt_data/my_scene" \
  --local_scene_dir "/content/my_scene" \
  --output_scene_dir "/content/drive/MyDrive/vggt_outputs/my_scene_ba" \
  --clean_local \
  --use_ba \
  --max_query_pts 2048 \
  --query_frame_num 5
```
