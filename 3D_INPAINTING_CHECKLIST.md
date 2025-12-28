# 3D Inpainting 准备清单

## ✅ 已完成的组件

### 1. **3D UNet架构**
- ✅ `models/guided_diffusion_modules_3d/unet.py` - 3D UNet for guided diffusion
- ✅ `models/sr3_modules_3d/unet.py` - 3D UNet for SR3
- ✅ `models/guided_diffusion_modules_3d/nn.py` - 3D utility functions

### 2. **3D Network支持**
- ✅ `models/network.py` - 支持3D diffusion forward/reverse steps
- ✅ 自动检测3D模块（通过`module_name`判断）
- ✅ 正确处理3D tensor的broadcasting

### 3. **3D Dataloader**
- ✅ `data/dataset_3d.py` - `Inpaint3DDataset`类
- ✅ 支持NIfTI文件读取（`.nii.gz`, `.nii`）
- ✅ 从CSV读取bounding box信息
- ✅ 支持多个bounding box合并为单个mask
- ✅ 支持两种UID格式提取
- ✅ CT normalization（nnUNet风格和简单normalization）

### 4. **工具函数**
- ✅ `data/util/nifti_3d_mask.py` - 3D mask生成
- ✅ `data/util/ct_normalization.py` - CT normalization函数

### 5. **模型和可视化支持**
- ✅ `models/model.py` - 支持3D数据检测和处理
- ✅ `core/logger.py` - 支持3D NIfTI文件保存
- ✅ TensorBoard可视化（提取中间slice显示）

### 6. **配置文件**
- ✅ `config/inpainting_3d_example.json` - 示例配置文件

## 📋 使用前需要准备的事项

### 1. **数据准备**
- [ ] 准备NIfTI图像文件（`.nii.gz`或`.nii`格式）
- [ ] 准备CSV文件，包含以下列：
  - `SeriesInstanceUID` - 图像UID
  - `new_coord_x`, `new_coord_y`, `new_coord_z` - bounding box中心坐标
- [ ] 确保CSV中的UID与NIfTI文件名匹配

### 2. **配置文件设置**
- [ ] 更新`config/inpainting_3d_example.json`中的路径：
  - `data_root`: NIfTI文件目录
  - `csv_path`: CSV标注文件路径
- [ ] 选择normalization方法：
  - `"normalization": "nnunet"` - 推荐用于CT/MRA
  - `"normalization": "simple"` - 简单HU范围normalization
- [ ] 设置UNet参数（根据你的数据尺寸调整）：
  - `image_size`: 建议设置为数据的最小维度
  - `in_channel`: 2（y_cond + y_noisy）
  - `out_channel`: 1
  - `inner_channel`: 基础通道数
  - `channel_mults`: 通道倍数

### 3. **训练参数调整**
- [ ] `batch_size`: 建议设为1（3D数据内存占用大）
- [ ] `num_workers`: 根据CPU核心数调整
- [ ] `log_iter`: 日志记录频率
- [ ] `val_epoch`: 验证频率

### 4. **内存和硬件**
- [ ] 确保GPU内存足够（3D数据需要更多内存）
- [ ] 考虑使用`use_checkpoint=True`来节省内存
- [ ] 如果内存不足，可以减小`image_size`或使用patch-based训练

### 5. **依赖检查**
- [ ] 安装所有依赖：`pip install -r requirements.txt`
- [ ] 确保安装了`nibabel`和`pandas`

## 🔍 关键注意事项

### Normalization
- **nnUNet normalization**: 输出Z-score范围（通常[-3, 3]），不是[-1, 1]
- 代码已自动处理这个差异，在可视化时会将Z-score映射到[0, 1]

### 数据维度
- NIfTI文件内部格式：`(X, Y, Z)` = `(width, height, depth)`
- UNet输入格式：`(C, D, H, W)` = `(1, Z, Y, X)`
- 代码会自动进行permutation转换

### Mask生成
- 每个NIfTI文件的所有bounding box会合并为一个mask
- Mask大小随机：10-30 pixels（可配置）
- Mask形状：长方体（cuboid）

### 可视化
- TensorBoard会显示3D volume的中间slice
- 保存的结果是完整的3D NIfTI文件
- 可以使用医学图像查看器（如ITK-SNAP）查看结果

## 🚀 开始训练

1. 准备数据和CSV文件
2. 更新配置文件
3. 运行训练：
   ```bash
   python run.py -p train -c config/inpainting_3d_example.json
   ```

## 📝 测试脚本

- `test_3d_dataloader.py` - 测试dataloader
- `test_3d_dataloader_single_uid.py` - 测试单个UID的可视化
- `test_3d_unet.py` - 测试3D UNet架构
- `test_3d_inpainting.py` - 测试3D inpainting流程

