# OmniScene 数据集实验文档

更新日期：2026-09-25。状态：**方案已获审阅并实施；已进行 CPU 协议测试和真实数据 GPU 冒烟检查，尚未开展正式训练／完整评估。** 运行方法及验证边界见第 11 节。

本实验在 `comp_svfgs` 分支将 UniSplat 适配为单帧六路环视重建方法，与 SVF-GS 使用相同的 OmniScene 数据、目标视角和指标口径。第 1～10 节保留已确认的设计依据；新增配置和入口现已实现，第 11 节记录实际路径、兼容性修正和检查结果。

## 1. 项目现状与已确认的实验边界

### 1.1 不能仅修改分辨率和训练轮数

论文确实报告了 nuScenes 实验，并说明沿用 Omni-Scene 的 bin 划分和目标视角协议；但这不等于当前仓库已经提供相应数据接口。论文实验与当前公开实现的覆盖范围不同。[论文实验设置](https://arxiv.org/html/2511.04595v1#S4.SS1)

实施前检查时，UniSplat 的 `main` 和 `comp_svfgs` 均指向 `3f7c245`，两分支之间没有代码差异。实施前仓库只有 Waymo 数据加载器、三阶段训练配置和 demo，没有 OmniScene 加载器、对应训练配置或完整的定量评估入口。当前提交还包含本地环境调整，不能把它描述成未经修改的原作者发布快照。

原实现依赖 LiDAR 深度、天空掩码、历史体素特征和历史高斯。适配本实验需要新增数据接口和评估流程，并明确关闭不符合边界的路径，而不只是改 YAML。依据：[训练入口](/home/dzp62442/Projects/UniSplat/train.py)、[Waymo 加载器](/home/dzp62442/Projects/UniSplat/dataset/waymo.py)、[Gaussian head](/home/dzp62442/Projects/UniSplat/model/gaussian_head/head.py)。

对照代码版本：SVF-GS `main@af39b31`；DepthSplat `comp_svfgs@405b9a5`。本文以这些版本的实际执行代码为准，旧文档中的不同描述不作为协议依据。

### 1.2 已由用户确认的选择

| 项目 | 本实验决定 |
| --- | --- |
| 模型规格 | 保留 UniSplat 官方入口默认结构：π³ large decoder；不改成 π³ base。UniSplat 没有 RE10K／Base 实验模板 |
| 初始化 | 加载原始 π³ 和 DINOv2 基础权重；UniSplat 新增模块随机初始化；不加载 Waymo UniSplat 权重 |
| 两种实验 | 112×200、224×400，均为 H×W；分别从基础权重独立训练，互不继承训练结果 |
| 几何路径 | 保留 π³ 预测几何和三阶段流程，用六路输入的 Metric3D 尺度深度替代 LiDAR 深度监督 |
| 总预算 | 每个分辨率 100,001 次优化器更新；三阶段分别 44,445／33,334／22,222 次 |
| 时序 | 禁止历史图像输入、历史特征、高斯缓存、跨样本状态和时序位姿补偿 |
| 动态掩码 | 开启；按 SVF-GS 的实际代码屏蔽新视角训练损失；关闭原 UniSplat 动态 BCE |
| 天空 | 关闭天空专用高斯分支，所有有效原图像素走普通分支；保留 400m 深度上限及网络输入通道数 |
| 尺寸适配 | 网络内部仅向右补边到 112×210／224×406，生成高斯前排除补边位置；目标渲染与评估保持完整原图 |
| Batch size | 训练、验证、mini 测试、完整测试均为 1；梯度累积为 1 |
| 验证 | 全局每 1,000 次优化器更新一次 |
| 训练中测试 | 第二、三阶段按全局每 10,000 步 mini 测试；第一阶段跳过；最终 100,001 步必须另做一次 mini 测试 |
| 日志与通知 | 本地日志；如启用 W&B，只能 offline；训练开始、每次 mini 测试完成后调用 `send_feishu` |

允许读取的模型相关数据只有六路当前 RGB、相机内外参、Metric3D 尺度深度和动态掩码。目标 RGB／相机／掩码用于监督和评估；DepthAnything V2 相对深度只用于 PCC。禁止读取或生成额外的 LiDAR、天空分割、语义、运动速度、光流等离线资产。本方案也不读取 Metric3D 的 `_conf.npy` 侧文件；深度有效性由已有深度的有限值和范围判断得到。

**“不使用时序信息”约束重建输入及模型状态，不删除用于监督／评估的 12 个新视角。** 这 12 个视角来自数据集同一 bin 内的其他采集位置；它们不进入 π³、DINOv2 或高斯生成网络。每个 bin 只重建一次高斯，再渲染全部 18 个目标视角。相邻训练样本仍可出现在同一训练集中，但不能共享特征或高斯。

## 2. 数据加载与 DepthSplat 的复用边界

### 2.1 数据位置和划分

默认数据根目录使用已存在的 `/home/B_UserData/dongzhipeng/Datasets/dataset_omniscene`。DepthSplat 的 `datasets/omniscene` 当前是指向该目录的软链接；UniSplat 无需重新生成一份数据。

| 用途 | 清单及选择规则 | 本次实际清单数量 |
| --- | --- | ---: |
| 训练 | `interp_12Hz_trainval/bins_train_3.2m.json`，全部 bins | 135,932 |
| 训练中验证 | `bins_val_3.2m.json` 的 `bins[:30000:3000][:10]` | 10 |
| mini 测试 | `bins_val_3.2m.json` 的 `bins[0::14][:2048]` | 2,048 |
| 完整测试 | `bins_val_3.2m.json`，不截断 | 30,080 |

这些是本次读取清单得到的数量；启动实验时重新检查清单、唯一 token 数和文件哈希。已抽查一个验证 bin 的 RGB、K、Metric3D 和 DA V2 文件，原始小图资产为 224×400；尚未逐文件审计整个数据集。

完整测试采用 SVF-GS 的 `total` 语义；另提供独立 `test` 入口，正式对比时使用。训练结束自动执行的是 mini 测试，不自动把 2,048 个 bin 的结果标成完整测试，也不擅自替换成 `center150`。依据：[SVF-GS 数据集](/home/dzp62442/Projects/SVF-GS/data/omniscene_dataset.py)、[DepthSplat 数据集](/home/dzp62442/Projects/depthsplat/src/dataset/dataset_omniscene.py)。

### 2.2 六路输入与十八路目标的严格顺序

相机顺序固定为：`CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT`。

1. 每个相机读取 `bin_info['sensor_info'][camera][0]`，组成六路中心输入。
2. 按上述相机顺序，为每个相机依次读取索引 `[1, 2]`，组成前 12 路新视角。
3. 在后面追加原六路中心输入，组成完整 18 路目标。

因此 `all_18 = target[0:18]`、`novel_12 = target[0:12]`、`input_6 = target[12:18]`。第三组可作为内部核对项，前两组都必须正式报告。训练也使用全部 18 个目标，不随机减少新视角。不能把目标顺序改成“先六路输入”，也不能把前后三相机的特殊输入采样规则误认为 12 路目标的生成规则。

读取现有 `bin_infos_3.2m/*.pkl` 的相机元数据即可；不读取 `LIDAR_TOP` 帧数来决定目标，也不加载点云。相机的 `sensor2lidar_transform` 名称表示已有标定参考坐标系，使用这个外参矩阵不等于输入 LiDAR 测量。代码内部可命名为 `camera_to_reference`，保留原有米制尺度和坐标，不做 baseline=1 归一化或 OpenCV/OpenGL 轴翻转。

### 2.3 资产和预处理

| 数据 | 现有资产路径模式 | 处理及使用阶段 |
| --- | --- | --- |
| RGB | `samples_small`／`sweeps_small` 下 `.jpg` | RGB、float32、[0,1]；按 SVF-GS 的图像加载／resize 方式到目标 H×W |
| 相机 K | `samples_param_small`／`sweeps_param_small` 下 `.json` | 使用与 224×400 资产配套的 K；resize 后分别缩放 `fx,cx` 与 `fy,cy` |
| 相机外参 | bin 内相机 `sensor2lidar_transform` | camera-to-reference，六路输入和 18 路目标使用同一参考系 |
| Metric3D | `samples_dptm_small`／`sweeps_dptm_small` 下 `*_dpt.npy` | float32、米制 Z 深度；需要 resize 时使用与 SVF-GS 相同的 PIL bilinear；只加载六路输入，供阶段 1/2 对齐 |
| 动态掩码 | `samples_mask_small`／`sweeps_mask_small` 下 `.png` | 灰度／255，白色保留、黑色屏蔽；bilinear resize 后保留浮点边界权重；后六路输入目标用全 1 |
| DA V2 | `samples_dpt_small`／`sweeps_dpt_small` 下 `.npy` | 仅 mini／完整评估时加载 18 路，供 PCC；转换见第 7 节 |

本实验不使用训练数据增强、随机裁剪或额外的基线尺度归一化。图像 resize 时同步调整 K，不对米制深度值乘缩放因子。输入源文件是共享已有资产，内部补边只发生在网络前向中，不写回数据集。

射线采用相机 Z 深度约定：`d_cam = K^-1 [u+0.5,v+0.5,1]^T`，`d_ref = R d_cam`，`o_ref = t`，点为 `o_ref + Z*d_ref`，**不将 d 归一化为单位向量**。应直接核对当前 `get_ray_directions`／`get_rays(normalize=False)` 的约定，避免把 Z 深度误作射线距离。

### 2.4 能复用什么，不能直接复制什么

| 项目 | DepthSplat 当前方式 | UniSplat 适配方式 |
| --- | --- | --- |
| bin、相机及目标选择 | 六路中心输入，12+6 目标 | 可以复用算法和顺序 |
| 资产路径解析／DA V2 转换 | `utils_omniscene.py` | 抽取到本仓库独立工具，不在运行时跨仓库 import |
| Metric3D | 当前 OmniScene loader 不加载 | 补充六路输入尺度深度，参考 SVF-GS |
| 动态掩码 | resize 后 `.bool()` | 改用 SVF-GS 的 float32 mask；不能丢弃灰度边界 |
| 内参 | 按宽高归一化 K | UniSplat 主路径使用像素 K；只在原深度对齐函数内部显式归一化 |
| batch | `context`／`target` | 通过 adapter 转为 π³ 图像及 Gaussian head 所需数据 |
| 像素尺寸 | `apply_patch_shim` 按 `4×4=16` 整除要求中心裁剪 | 不复用此 shim；保持完整视场，内部按 patch 14 补边 |
| 测试子集 | 当前 `stage='test'` 固定 `[0::14][:2048]` | 显式 `mini`／`total`，不可继续隐藏截断 |
| near／far | 默认 0.5／100 | 使用 UniSplat 渲染器 0.1／1000；几何深度另截断至 400m |

DepthSplat 当前 112×200 实际会经过 shim 变成 112×192；224×400 可整除 16，不发生这一裁剪。本项目按用户要求评估完整 112×200，不能直接把已有裁剪结果当作完全相同的像素协议。若后续使用 DepthSplat 旧结果进对比表，需核对实际渲染尺寸并统一协议后再比较；本轮不修改 DepthSplat。

依据：[DepthSplat 资产加载](/home/dzp62442/Projects/depthsplat/src/dataset/utils_omniscene.py)、[patch shim](/home/dzp62442/Projects/depthsplat/src/dataset/shims/patch_shim.py)、[encoder shim 调用](/home/dzp62442/Projects/depthsplat/src/model/encoder/encoder_depthsplat.py:379)、[SVF-GS 资产加载](/home/dzp62442/Projects/SVF-GS/data/transforms/loading.py)。

## 3. 无时序模型路径和尺寸适配

拟将重建与目标渲染明确分开，数据流为：

```text
六路当前 RGB + K + 外参
  → 网络内右侧补边
  → 冻结 π³：当前六视角特征、局部点图 Z
  → scale/shift 对齐（训练阶段 2 用 Metric3D 对齐值，其余重建用预测值）
  → DINOv2 + 深度/RGB/射线 embedding + 图像特征解码
  → 排除补边位置，构造当前帧稀疏体素
  → 当前帧稀疏 U-Net 与空间融合
  → 点锚定高斯分支 + 体素高斯分支 → 最终高斯
  → 使用 18 路目标相机渲染 RGB/Z 深度 → 监督或指标
```

阶段 1 只执行 π³ 和尺度预测／监督部分，不要求生成可评估的高斯。Metric3D 不是阶段 3 或推理阶段的直接几何锚点；目标 RGB、目标 DA V2 和目标掩码绝不进入 `reconstruct(context)`。

### 3.1 彻底关闭跨样本状态

新增 `temporal_enabled: false` 的明确分支，同时覆盖训练、验证和测试。该分支不调用 `history_queue.get/set`，不做历史位姿变换、历史高斯拼接、动态分数驱动的记忆筛选或未来视角动态投影掩码。

在 U-Net 中仅保留当前帧空间特征与融合卷积；绕过历史稀疏张量相加。原当前帧位置 embedding 可以保留；当前类型 embedding 若保留，只是固定的当前帧类别，不接收帧号／时间差。`scene_id`、`bin_token` 只用于记录，不能参与网络重建。原按场景连续组织的 Waymo sampler 改为普通可复现的随机 bin sampler。

仅给每个 batch 清空缓存不足以作为最终实现：还要取消队列写入及当前 demo 中历史高斯补全路径。后续验收应检查同一个 bin 单独执行、在其他 bin 后执行、打乱顺序执行，结果在数值容差内一致。

### 3.2 没有天空掩码时的处理

新增 `use_sky_branch: false`，拆开原先混用的 `sky_mask: 400.0` 含义：以 `depth_max_m: 400.0` 表示数值截断，以开关控制天空分支。

所有原图像素按普通点锚定分支生成高斯，不把任何深度阈值伪装成天空标签。原五通道深度 embedding 仍为 `[Z/400, 常数1, RGB]`；常数通道不从外部加载。补边有效性由原图 H×W 所定义的切片域表示（无需额外加载 mask），不充当天空信息。

为保留原模块结构与权重键，天空 MLP 可继续注册但冻结且不执行；参数报告必须单列此类禁用模块。动态输出通道同样保留原形状，但不计算 BCE、不参与高斯保留／剔除。不能根据未训练的动态分数筛掉当前帧动态高斯。

### 3.3 patch 14 与原图分辨率

| 项目 | 小分辨率 | 大分辨率 |
| --- | --- | --- |
| 数据／目标／指标 H×W | 112×200 | 224×400 |
| 网络内部 H×W | 112×210 | 224×406 |
| 右侧补边 | 10 列 | 6 列 |
| patch 网格 | 8×15 | 16×29 |

RGB 使用右侧 replicate padding；已有深度／特征等在需要组成网络输入时作相应补边，同时生成原始尺寸有效域。只向右补边，原有像素坐标、像素 K 和外参不变；有归一化 K 的局部计算必须使用所在张量的实际宽高。

π³、DINOv2、Plücker/depth PatchEmbed 和图像特征解码使用补边画布。尺度对齐仅统计原图有效域；先裁回原图再做 32×32 对齐，不能让复制的边界深度重复充当监督。构造点锚和体素前剔除补边像素，不能让复制边缘产生额外高斯。体素投影采样时必须明确特征图对应的是补边画布还是裁回的原图，并使用一致的 K、宽高和采样坐标。

原 head 用单一 `H,W` 同时处理特征、像素点和目标渲染，必须拆成 `encoder_image_shape` 与 `image_shape`，不能只在入口 `pad()`。渲染器始终使用原图尺寸和 K，损失与 PCC 永远不统计补边。完整投影矩阵保留主点偏移，禁止假定补边后 `cx=W/2`。

原 `cfg.resolution` 在 Waymo/head 中采用 `[W,H]`；新配置对外统一 `[H,W]`，只在旧接口边界转换，避免把 `[112,200]` 原样传给 `render_w, render_h = ...`。

## 4. 配置组织、模型参数和加载规则

### 4.1 拟新增配置

```text
configs/
  dataset/omniscene.yaml                # 根目录、split、相机/目标、资产协议
  model/unisplat_static.yaml            # 原 UniSplat 模型参数及静态开关
  train/unisplat_three_stage.yaml       # 三阶段预算、优化器、评估调度
  experiment/omniscene_112x200.yaml      # 小分辨率实验入口
  experiment/omniscene_224x400.yaml      # 大分辨率实验入口
```

沿用 UniSplat 的 argparse + OmegaConf，不引入 DepthSplat 的 Hydra/Lightning 训练框架。新增配置加载 helper，按 `includes` 给出的顺序加载基础文件，再用实验文件和 CLI dotlist 覆盖；所有 include 路径相对于 UniSplat 根目录解析。现有 `OmegaConf.load(args.config)` 不自动支持这套组合，需要显式实现并校验，不能只添加 YAML 的 `defaults` 期待其生效。

小分辨率实验入口草案如下；大分辨率文件只改实验名和 `Dataset.image_shape: [224,400]`：

```yaml
includes:
  - configs/dataset/omniscene.yaml
  - configs/model/unisplat_static.yaml
  - configs/train/unisplat_three_stage.yaml
Experiment:
  name: unisplat_omniscene_static_112x200
  seed: 42
Dataset:
  root: /home/B_UserData/dongzhipeng/Datasets/dataset_omniscene
  image_shape: [112, 200]       # H, W
  num_context_views: 6
  num_target_views: 18
  train_batch_size: 1
  val_batch_size: 1
  test_batch_size: 1
  use_dynamic_mask: true
Model:
  pi3_decoder_size: large
  pi3_ckpt: ckpt/pi3/model.safetensors
  dinov2_ckpt: ckpt/dinov2/dinov2_vits14_reg4_pretrain.pth
  temporal_enabled: false
  use_sky_branch: false
  padding: right_replicate_to_14
Train:
  max_steps: 100001
  stage_steps: [44445, 33334, 22222]
  gradient_accumulation_steps: 1
  val_every_steps: 1000
  mini_test_every_n_val: 10
  mini_test_stages: [2, 3]
  final_mini_test: true
Evaluation:
  split: mini
  view_groups: [all_18, novel_12, input_6]
  metrics: [psnr, ssim, lpips, pcc]
  pcc_reference: depth_anything_v2
  depth_semantics: accumulated_z
  pixel_protocol: full_image
Logging:
  backend: local
  wandb_mode: offline
Feishu:
  enabled: true
  events: [train_start, mini_test_complete]
```

训练根目录按分辨率隔离。保存完整 `resolved_config.yaml`、三个仓库参考 SHA、输入清单及其哈希、基础权重路径／哈希、阶段和全局步数。参数冲突、阶段预算不等于 100001、非 6/18 视角或启用禁止的数据源应在启动前报错。基础权重必须核对键名、形状和加载范围，禁止静默缺失 π³／DINOv2 参数后继续随机初始化。原 Waymo 配置保留，新增实验从共享 helper 读取自身配置。

### 4.2 模型结构与具体参数

此处的 encoder、几何 decoder 和 Gaussian decoder 都取自 UniSplat；不能套用 DepthSplat 的 depth candidates、cost-volume U-Net、RE10K MSE 权重等配置。

| 部分 | 本实验使用的 UniSplat 配置 |
| --- | --- |
| π³ encoder | `dinov2_vitl14_reg`，patch=14，特征宽度 1024；加载 π³ 权重后全程冻结 |
| π³ decoder | `large`，36 blocks，dim=1024，16 heads，MLP ratio=4，RoPE100，5 个特殊 token |
| π³ 输出 | 保留原局部点图和中间特征；以局部点图 Z 为几何预测，不采用预测相机替代已知标定 |
| Gaussian head | `GuassianHead(dim_in=2048, patch_size=14)`；实际中间特征索引 `[3,8,12,16]`，是成对拼接后的 π³ 中间输出索引 |
| 图像分支 | DINOv2 ViT-S/14 reg4，dim=384，投影到 2048；初始化预训练权重，阶段 2/3 可训练 |
| 多尺度图像 decoder | 输出通道 `[256,512,1024,1024]`；融合宽度 256；体素图像特征投影到 32 |
| 射线／深度 embedding | Plücker 6 通道、深度/RGB 5 通道，均 patch=14、dim=1024；组合后投影到 2048 |
| 尺度预测 decoder | Gaussian head 自己的 `point_decoder`：2048→1024、16 heads；与冻结的 π³ point decoder 是不同模块 |
| scale／shift MLP | 各为 1024→1024→1024→1；scale 最后 exp 保证为正 |
| 当前帧体素范围 | `[-72,-72,-4,72,72,12]` 米 |
| 初始体素大小 | `[0.1,0.1,0.2]` 米；保留原稀疏 U-Net 的分辨率层级 |
| 体素 U-Net | 输入 XYZ+RGB+32 维图像特征，共 38 维；通道 32/64/128/256，输出特征 64；保留空间融合卷积 |
| 双分支 | 点锚定分支和体素生成分支均保留；`voxel_gs_num=2`；原参数输出 15 维，最后动态分数不用于本实验 |
| 点高斯参数 | `offset_scale=2.0`，`max_scale=0.1`，sigmoid RGB/opacity、softplus scale、归一化 quaternion |
| 体素高斯参数 | `max_scale_voxel=0.2`；其他激活沿用原代码 |
| 渲染器 | `GaussianRenderer_dyn`，黑背景，直接 RGB／SH degree 0，near=0.1、far=1000；图像尺寸取原始目标 H×W |
| 深度数值上限 | `depth_max_m=400.0`；与渲染器 far 不同，不套用 DepthSplat 的 100m |

历史保留比例、历史点数上限、`dyn_save_thre`、天空专用尺度等在静态实验中不生效；不把这些数值误列为本实验高斯容量预算。

结构依据：[π³](/home/dzp62442/Projects/UniSplat/pi3/models/pi3.py)、[Gaussian head](/home/dzp62442/Projects/UniSplat/model/gaussian_head/head.py)、[稀疏 U-Net](/home/dzp62442/Projects/UniSplat/model/gaussian_head/unet.py)、[渲染器](/home/dzp62442/Projects/UniSplat/model/layers/gaussian_dyn.py)。

## 5. 三阶段训练、监督与损失

### 5.1 阶段预算和学习率

这里的 `global_step` 表示已经完成的优化器更新数，初始为 0，最终恰好为 100001。不是每个阶段分别训练 100001 步，也不沿用可能多跑一步的 `<= max_steps` 循环定义。

| 阶段 | 全局更新范围（含端点） | 更新次数 | 几何尺度来源 | 可训练部分 | 最大 LR |
| --- | --- | ---: | --- | --- | --- |
| 1 | 1–44,445 | 44,445 | Metric3D 对齐值作为监督；训练预测 scale/shift | Gaussian head 的 point decoder、scale head、shift head | 1e-4 |
| 2 | 44,446–77,779 | 33,334 | 训练前向用 Metric3D 对齐的 scale/shift | DINOv2 图像分支及其余有效 Gaussian head；尺度模块冻结 | DINOv2 1.5e-5，其余 1.5e-4 |
| 3 | 77,780–100,001 | 22,222 | 冻结尺度模块的预测 scale/shift | 与阶段 2 相同 | DINOv2 1.5e-5，其余 5e-5 |

π³ 始终冻结。原阶段 3 的 YAML 注释虽有“whole head”字样，实际 `apply_stage_freeze` 仍冻结三个尺度模块；本方案遵守实际代码。阶段 2、3 的 mini／完整评估都走预测尺度，绝不能用 Metric3D 对齐值做评估时的几何校正。

各分辨率只在阶段 1 初始化基础模型；阶段 2 完整继承同分辨率阶段 1 的模型状态，阶段 3 继承阶段 2。阶段切换重建优化器和本阶段 OneCycleLR；必须显式重设 `requires_grad`，不能在同一对象上简单连续调用当前冻结函数，否则阶段 1 冻结的图像／高斯模块不会自动解冻。LPIPS 等损失网络始终冻结且保持 eval。

原配置依据：[stage 1](/home/dzp62442/Projects/UniSplat/configs/waymo_stage1.yaml)、[stage 2](/home/dzp62442/Projects/UniSplat/configs/waymo_stage2.yaml)、[stage 3](/home/dzp62442/Projects/UniSplat/configs/waymo_stage3.yaml)、[冻结逻辑](/home/dzp62442/Projects/UniSplat/train.py:71)。

### 5.2 监督真值的具体来源

**阶段 1：尺度伪真值。** 六路输入的 Metric3D 尺度深度替代原 `single_depthmaps` 中的 LiDAR 投影。有效范围为有限且 `0.1 < Z < 400`，不使用目标深度、额外置信度或天空标签。

沿用原 `depth_to_points`、`mask_aware_nearest_resize(size=(32,32))` 和 `align_points_scale_z_shift`。对齐权重使用原有有效 mask / GT Z，`trunc=1.0`；求得每个相机的 scale 和 Z-shift，detach 后监督尺度预测模块。不是自行改成简单深度均值比例，也不是把 Metric3D 直接输入高斯 MLP。无有效对齐的相机不参与尺度损失；需记录无效数，防止训练悄悄全为零。

**阶段 2：尺度教学 + RGB 重建。** 用同一对齐函数得到的 scale/shift 修正 π³ 局部 Z，训练其余高斯生成网络。尺度值不反向传播。图像监督是 18 路目标 RGB，辅助体素重建监督为后六路中心 RGB。

**阶段 3：预测尺度 + RGB 重建。** 不依赖 Metric3D 读取或对齐；沿用阶段 2 的图像损失，前向几何改用冻结的尺度模块预测。独立测试同样不需要 Metric3D。

### 5.3 损失项及权重

| 损失 | 阶段 | 数学／实现定义 | 权重 |
| --- | --- | --- | --- |
| scale | 1 | 有效相机上的 `abs(pred_scale - aligned_scale)` 平均 | 0.1 |
| shift | 1 | 有效相机上的 `abs(pred_shift - aligned_shift_z)` 平均 | 1.0 |
| RGB 重建 | 2/3 | `mean(((pred_rgb - gt_rgb) * M)^2)` | 外层 5.0，每视角 1.875 |
| 感知损失 | 2/3 | UniSplat 自带 VGG LPIPS loss，输入分别乘相同 M 后计算 | 外层 0.05，每视角 0.625 |
| 体素辅助 RGB | 2/3 | 仅体素分支渲染后六路中心相机；平方误差乘原体素范围内像素 mask 后全图平均 | 0.25 |
| 动态 BCE | 不使用 | 用户确认关闭，不用伪造全零动态真值维持该项 | 0 |

原 `l1_loss_mask` 名字容易误导：实际重建误差是平方误差。将原 10 项数组扩展为 **18 项相同值**：`l1_loss_mask=[1.875]*18`、`p_loss_mask=[0.625]*18`，保留原每视角系数和 mean reduction。对应全局等效系数分别为 9.375 与 0.03125；不能只抄外层 5／0.05 而忽略内层权重。

不新增深度渲染监督、SSIM、PCC、光流、法线、天空分类或其他训练损失。测试 LPIPS 使用通用 VGG 指标实现，与训练的自带 LPIPS loss 分开管理；不为了统一名字而替换训练损失实现。

### 5.4 动态掩码必须与当前 SVF-GS 的实际行为一致

令 `M[:12]` 为新视角文件提供的浮点有效 mask，`M[12:18]=1`。训练 RGB 和 LPIPS 都分别对预测与真值乘 M，然后使用原损失的平均方式；**不按有效像素数量重新归一化**。浮点 mask 下 RGB 平方损失中的权重是 `M²`，不能替换为 `M*(pred-gt)²`。

LPIPS 在原始目标分辨率计算，掩码与图像同尺寸；不沿用原 UniSplat “RGB masked、LPIPS unmasked”的行为。原动态点投影得到的额外 mask 也关闭，避免与 SVF-GS 的目标 mask 联集后改变监督范围。中心六路不屏蔽，因此体素辅助重建保持原方式。

此掩码只改变训练／验证损失。mini 和完整评估的 PSNR、SSIM、LPIPS、PCC 全部在完整目标图像上计算，不使用动态掩码，也不根据渲染 opacity 缩减评估区域。

当前 SVF-GS 的 `AGENTS.md` 对 mask 的部分描述与执行代码不一致，本协议明确依据 [load_conditions](/home/dzp62442/Projects/SVF-GS/data/transforms/loading.py:109) 和 [compute_loss](/home/dzp62442/Projects/SVF-GS/model/omni_gs.py:446)。

### 5.5 优化器、精度和保存

沿用 AdamW：初始 `betas=(0.9,0.95)`、`eps=1e-8`、`amsgrad=False`，weight decay=0.01；bias 和一维参数不做 weight decay。LR 分组先匹配 `gaussian_head.image_backbone`，再匹配 `gaussian_head`，避免 DINOv2 被宽泛组覆盖。只能收集本阶段 `requires_grad=True` 的有效参数。

每阶段独立 OneCycleLR，以该阶段更新次数作为 `total_steps`，`max_lr` 为上表，`pct_start=0.01`、`anneal_strategy='cos'`；其余参数显式固定为本地 PyTorch 源码中原入口使用的默认值：`div_factor=25`、`final_div_factor=10000`、`three_phase=False`、`cycle_momentum=True`、`base_momentum=0.85`、`max_momentum=0.95`。因此各组初始 LR 为 `max_lr/25`，最低目标 LR 为 `max_lr/250000`。OneCycleLR 会调节 Adam 的一阶动量，不能将初始化 betas 误写成全过程不变。梯度范数裁剪为 10，随机种子 42。DataLoader 默认沿用 `num_workers=8`、`pin_memory=True`，验证／测试不 shuffle。

保留当前精度策略：冻结 π³ 在 `no_grad` 和支持时的 BF16 autocast 下运行，其余 Gaussian head（含 DINOv2）在 FP32 路径中运行，沿用 GradScaler 机制。若发生非有限梯度且更新被跳过，不推进“已完成优化器更新数”或 scheduler，记录事件并排查；不能以失效的 batch 消耗有效训练预算。

按单 GPU、batch=1、无梯度累积设计主实验。若扩展多 GPU，必须另行明确有效 batch；不能把每卡 batch=1 表述成总 batch=1。并行训练不能沿用绕过 DDP wrapper 直接调用可训练 head 的方式，须确保完整可训练前向受 DDP 管理。224×400 的显存和速度需要实现后的实测，本轮没有验证能否在具体 GPU 上训练。

每次验证点保存可恢复状态，阶段边界和最后一步另存固定 checkpoint。每个 checkpoint 同目录原子保存模型、optimizer、scheduler、scaler、RNG、sampler 游标、`global_step/stage/stage_step/val_count/last_mini_test_step` 和配置身份，不能沿用旧的“按 epoch 找权重、另读一个全局 training_state”的不配对恢复方式。

## 6. 主程序、验证／测试调度及通知

### 6.1 与 DepthSplat 主程序的区别

DepthSplat 的主路径是 Hydra 配置 → Lightning DataModule／ModelWrapper → `encoder(context)` 得到其 `Gaussians` 数据类 → decoder 渲染 target。UniSplat 当前是 argparse/OmegaConf → Waymo loader → π³ → 挂载的 Gaussian head，head 内含点／体素处理、渲染和损失。

因此可以复用清单选择、指标函数和纯汇总代码，不能直接调用 DepthSplat 的 DataModule/ModelWrapper 或把其 `encoder(context)` 替换进来。需要本地 OmniScene adapter 和 UniSplat 自己的训练／评估调度。

拟新增 `train_omniscene.py`、`eval_omniscene.py` 和共享的构建／评估 helper；保留旧 `train.py`／`demo.py` 的 Waymo 用法。Dataset 返回独立的 `context`、`target` 与元信息，再由 adapter 对接原 head 的像素 K、c2w、射线等字段；训练阶段 1/2 的深度监督通过独立参数传递，不混入测试 `context`。

| 数据字段 | shape／含义 |
| --- | --- |
| `context.image` | `[B,6,3,H,W]`，只有中心六路 |
| `context.intrinsics`／`extrinsics` | `[B,6,3,3]` 像素 K／`[B,6,4,4]` c2reference |
| `supervision.input_metric_depth` | 阶段 1/2 的 `[B,6,H,W]`，测试不加载 |
| `target.image`／相机 | `[B,18,3,H,W]` 及 18 路 K、c2reference |
| `target.loss_mask` | `[B,18,H,W]` 浮点，仅训练／验证损失使用 |
| `target.rel_depth` | `[B,18,H,W]`，仅定量评估加载 |
| `meta` | bin token、scene id、视角身份、split、原始尺寸；不参与几何预测 |

公共模型接口拆成 `reconstruct(context, stage, supervision=None)` 与 `render(gaussians, target_cameras)`；训练 loss 和测试指标在调用层计算。目标相机可进入 renderer，目标图像、mask、DA V2 只能进入 loss／metric，不能进入高斯构建。既利于检查输入边界，也提供完整重建计时的准确终点。

### 6.2 精确调度

`global_step` 跨阶段累计，数据 epoch 只负责遍历／重洗牌，不决定停止或评估时间。1,000 步和 0.01 epoch 在本训练集大小下并不相等，本实验选择明确的每 1,000 步。

- 验证在 `1000, 2000, …, 100000`，共 100 次。每次遍历固定 10 个验证 bin，batch=1。阶段 1 验证 scale／shift；阶段 2 验证训练目标损失并标注使用对齐尺度；阶段 3 验证预测尺度路径的图像损失。验证值汇总全部 10 个 bin，不仅记录最后一个 batch。
- 每完成 10 次验证检查一次 mini 测试条件。阶段 1 的 10000／20000／30000／40000 仅记为跳过，不评估随机 Gaussian head。
- 实际周期 mini 测试步数为 **50000、60000、70000、80000、90000、100000**。阶段切换不重置验证计数。
- **100001 步完成后**，保存最终 checkpoint，再执行独立最终 mini 测试并落盘、通知。100000 步已测过也不能代替这次测试。若评估中断，恢复时继续／重做最终评估，不额外执行训练更新。

mini 评估使用 `eval()`／`inference_mode()`、预测 scale/shift、无历史状态，完成后恢复各模块正确模式（冻结 backbone、LPIPS 仍保持 eval）及训练 RNG 状态。评估不反向传播，不改 optimizer/scheduler/global step。阶段 2 的 mini 结果与阶段 3 使用同一真实推理路径，不混入对齐尺度结果。

完整测试通过 `eval_omniscene.py --split total` 单独启动，默认评估指定最终 checkpoint；完整测试集不参与训练中的模型选择。输出目录区分分辨率、checkpoint、step 和 split，禁止周期 mini 覆盖最终 mini。

### 6.3 日志与飞书

原 UniSplat 使用本地 Logger，没有必须迁移的 W&B 依赖。主实验保留本地文本／JSONL 日志；若实现时加入 W&B，显式 `mode=offline`，不调用在线初始化或自动同步。π³、DINOv2、训练和评估 LPIPS 权重须在正式训练前已可本地读取，不在训练循环触发临时下载。

参考 [SVF-GS 通知调用](/home/dzp62442/Projects/SVF-GS/trainer.py:25)，从配置给定的本地模块根目录导入 `auto_monitor.send_feishu.send_feishu`，默认可尝试 `/home/dzp62442/Libraries`。不复制 webhook／token 到仓库，也不在本轮文档编写时实际发送通知。

rank 0 在以下时机调用 `send_feishu(subject, content)`：

1. 训练开始：实验名、分辨率、静态输入协议、初始化来源、三阶段预算、输出目录、当前／恢复步数、参数统计。
2. 每次 mini 评估完整落盘后，包括最终评估：checkpoint/step、split、完成数量／预期数量、`all_18` 和 `novel_12` 四项指标、重建耗时、训练进度／ETA、结果目录。

通知通过独立 worker／有超时的调用发送，失败写入本地待重试日志，有限重试不阻塞训练；恢复按 `(run_id,event,step)` 去重。模块缺失在启动检查中明确报出，不能将“没有导入成功”记录成通知已发送。若指标集合不完整，通知标记未完成，不能发送正常完成结论。

### 6.4 拟提供的调用示例

以下入口及 CLI 参数将在审阅通过后实现；基础权重路径需替换为实际文件，正式训练不会自动下载或自动选择 Waymo 权重。

```bash
python /home/dzp62442/Projects/UniSplat/train_omniscene.py \
  --config /home/dzp62442/Projects/UniSplat/configs/experiment/omniscene_112x200.yaml \
  --pi3_ckpt /absolute/path/to/pi3.safetensors \
  --dinov2_ckpt /absolute/path/to/dinov2_vits14_reg4_pretrain.pth

python /home/dzp62442/Projects/UniSplat/train_omniscene.py \
  --config /home/dzp62442/Projects/UniSplat/configs/experiment/omniscene_224x400.yaml \
  --pi3_ckpt /absolute/path/to/pi3.safetensors \
  --dinov2_ckpt /absolute/path/to/dinov2_vits14_reg4_pretrain.pth

python /home/dzp62442/Projects/UniSplat/eval_omniscene.py \
  --config /home/dzp62442/Projects/UniSplat/configs/experiment/omniscene_112x200.yaml \
  --checkpoint /absolute/path/to/step_100001/model.safetensors \
  --split total
```

大分辨率完整测试使用对应大分辨率配置和 checkpoint。训练入口默认自动恢复当前实验目录下最新保存的 checkpoint，也支持 `--resume /absolute/path/to/checkpoint_dir` 手动指定；`--checkpoint` 只用于评估，不能与恢复训练混淆。

## 7. PCC 与两组质量指标

### 7.1 相对深度加载

复用 DepthSplat／SVF-GS 对现有 DA V2 文件的数学处理。文件中的原值按 disparity 使用：先将 disparity bilinear resize 到目标尺寸，再执行

```text
r = min(max(disp) / (min(disp) + 0.001), 50)
d_min = max(disp) / r
relative_depth = 1 / max(disp, d_min)
relative_depth = (relative_depth - min) / (max - min)
```

这一步是读取已有数据时的数值转换，不运行新的离线深度预测。训练和普通验证不加载 DA V2；mini／完整测试为 18 路目标加载。不得把 Metric3D 替代成 PCC 参考，也不得把 DA V2 当作网络输入或训练监督。

常数、空值或非有限参考图要明确记录。不能默默将其改成零并产生“有效 PCC”，也不能静默删掉 bin 后继续声称评估完整。

### 7.2 使用现有 UniSplat 渲染深度

UniSplat 的 `GaussianRenderer_dyn.render()` **已经返回 `depth`**，shape 为 `[V,1,H,W]`；只需在新的评估接口透传，整理为 `[B,18,H,W]`。不需要另加一个深度网络。

其 CUDA 实现先取高斯中心的相机 Z，再计算

`D(u) = Σ_i T_i(u) α_i(u) z_i`。

这是 `accumulated_z`，无前景贡献时为 0，**没有除以累计 alpha**。DepthSplat 的 `depth_mode='depth'` 同样把相机 Z 当作颜色、以黑背景进行 alpha 合成。PCC 主结果因此使用现有 accumulated Z，不切换成 expected depth，不取倒数，也不使用 π³ 输入深度替代目标视角渲染深度。

实现依据：[UniSplat 渲染返回](/home/dzp62442/Projects/UniSplat/model/layers/gaussian_dyn.py:250)、[CUDA 深度累积](/home/dzp62442/Projects/UniSplat/submodules/diff-gaussian-rasterization-feature/cuda_rasterizer/forward.cu:365)、[DepthSplat 深度渲染](/home/dzp62442/Projects/depthsplat/src/model/decoder/cuda_splatting.py:225)。

### 7.3 计算位置与聚合

测试循环在同一组最终高斯渲染完 18 路 RGB／深度后统一调用指标工具：

| 指标 | bin 内计算 | 跨 bin 汇总 |
| --- | --- | --- |
| PSNR | 图像限制到 [0,1]，逐视角 RGB MSE→PSNR；分别取 18／12 视角均值 | 每个 bin 等权平均 |
| SSIM | 与 DepthSplat／SVF-GS 相同的 skimage 实现，win_size=11、Gaussian weights、data_range=1 | 每个 bin 的组内均值再等权平均 |
| LPIPS | `LPIPS(net='vgg')`，`normalize=True`；逐视角，再取对应组平均 | 每个 bin 等权平均 |
| PCC | 对每个 bin 的相应视角组，将参考深度和渲染深度各展平为一个向量，计算一次 Pearson | 每个 bin 的组 PCC 等权平均 |

PCC 不能先逐视角计算再平均，也不能把 2,048／30,080 个 bin 的像素合并计算一次。`all_18` 和 `novel_12` 必须分别重新展平计算，不能用加权差从 all/input PCC 推出 novel PCC。

使用无跨调用累积状态的 Pearson 实现，或确保复用 TorchMetrics 时每次独立计算／正确 reset；与参考 `compute_pcc` 在非退化张量上做数值等价验证。零方差或非有限情形返回显式无效状态；汇总列出失败 token，并将完成状态设为 false，禁止用忽略 NaN 的平均得到貌似完整结果。

全部指标使用完整原图，无动态／天空／opacity 评估 mask。PCC 报告为相对深度一致性诊断；它使用 DA V2 伪参考，不应解释为绝对几何精度。

### 7.4 复用 DepthSplat 指标与统计时的修改

[DepthSplat metrics](/home/dzp62442/Projects/depthsplat/src/evaluation/metrics.py) 的 PSNR／SSIM／LPIPS／PCC 函数可以迁入本项目。本地 OmniScene 常规测试分支当前主要写整体 scores，不能直接视为已包含用户要求的两组结果。

[DepthSplat zero_shot.py](/home/dzp62442/Projects/depthsplat/src/evaluation/zero_shot.py) 已有 `view_group_records`、`summarize_records` 和完整性检查，可抽取通用部分。但它服务 PandaSet／DDAD，存在 `pcc_reference='metric3d_v2'` 及将 novel_12 标成 diagnostic 等特定元数据；本实验必须改成 `depth_anything_v2`，并将 all_18、novel_12 都作为正式报告组。不要连同零样本数据集分支／ego mask 配置整体复制。

输出至少包含：

- `per_bin_metrics.csv`：bin token、scene、split、checkpoint、step、view_group、四项指标、实际图像尺寸。
- `evaluation_summary.json`：两个必报视角组（可额外 input_6）、预期／完成 bin 数、缺失／重复／异常列表、完成状态、聚合定义。
- `data_provenance.json`：清单哈希、checkpoint 哈希、配置身份、`pcc_reference=depth_anything_v2`、`depth_semantics=accumulated_z`、`pixel_protocol=full_image`、无时序标记。
- `model_parameters.json`、`reconstruction_time.json`：第 8 节的参数和耗时统计。

如保留 `scores_pcc_all.json`／`scores_all_avg.json` 兼容输出，必须标明其中 `all` 指 all_18；两组结构化汇总仍是正式结果。每次测试重新初始化容器，防止不同时刻的 mini 结果串入同一个平均。

## 8. 参数量与完整重建耗时

### 8.1 参数量

在每个阶段设置冻结状态后统计、最终 checkpoint 加载后再次统计：

```text
trainable = Σ numel(p), p.requires_grad == True
frozen    = Σ numel(p), p.requires_grad == False
total     = trainable + frozen
```

主报告范围是注册的重建模型：π³、DINOv2、Gaussian head；不包含训练／评估 LPIPS 或其他指标网络，也不将运行时高斯个数、buffer 或 optimizer state 算成模型参数。共享参数只计一次。

π³ 中保留但未被下游使用的输出头、禁用且冻结的天空头仍属于注册模型，应计入 frozen／total 并单列 `disabled_or_unused` 明细，不能通过只统计可训练 head 隐藏基础模型规模。动态通道若与 RGB 等通道共享同一个 Linear 参数张量，不虚构逐通道的 `requires_grad` 参数数。另记录各阶段冻结状态，最终主表采用阶段 3 的 trainable/frozen/total。

两种分辨率预计结构参数数相同，但必须由实际构建后的模型验证；本轮不实例化大模型，因此不填写未经测量的参数数值。

### 8.2 计时边界

主计时 `reconstruction_ms` 从已加载 batch 开始送入重建流程计起，到最终点分支＋体素分支高斯全部生成、激活并合并完成为止，batch=1、每 bin 一次。它包括六路模型输入的 H2D 拷贝、内部补边／射线准备、完整 π³ 前向、DINOv2、scale/shift 预测、稀疏体素构建／处理和双分支高斯生成。

不包括磁盘读取／DataLoader 等待、checkpoint 加载、目标 RGB／DA V2 传输、18 视角最终渲染、PCC／LPIPS 计算、文件保存和消息推送。推理阶段不做 Metric3D 对齐。另报告 `network_reconstruction_ms`（输入已在设备上时的相同重建路径）和 H2D 分项，使与其他项目比较时能匹配边界，不能把纯 head 时间命名为完整重建时间。

开始和结束做 CUDA synchronize，用 `perf_counter` 统计含 CPU 控制／稀疏构建开销的墙钟时间；不能仅累计 CUDA event 就遗漏主机端必需操作。计时期间保持 eval／inference_mode、不写图、不执行通知。

先作 5 次独立 warmup 前向，再从第一个测试 bin 开始正式计时和计算指标；warmup 不排除任何 bin 的质量评估。保存逐 bin 时间、平均值、中位数、P95、样本数、GPU／软件环境／精度、原图和网络内部尺寸。两种分辨率独立记录，跨方法比较必须核对相同硬件、精度、输入和上述计时起止点。

## 9. 实施顺序和验收范围

审阅通过后，计划按以下顺序实现：

1. 配置组合／校验和 OmniScene loader：先确保六路输入、18 路目标、K、mask、split、DA V2 与参考项目一致。
2. 静态重建路径：拆分 reconstruction/render，关闭历史、天空和动态 BCE，完成双尺寸内部补边；用 Metric3D 接通阶段 1/2 对齐。
3. 三阶段优化器／恢复、验证／mini／final 调度、本地日志和 `send_feishu` 事件。
4. 原尺寸 RGB／深度评估、PCC 两组统计、完整性检查、参数量和重建计时。
5. 完成针对适配风险的验证，再按用户授权开展训练／正式评估。

后续需重点验收的行为：

- 两个配置都解析为 100001 总更新、batch=1、44,445/33,334/22,222 阶段预算；调度不在恢复后重复更新、遗漏最终测试。
- 与 SVF-GS 抽样比对相同 token 的 6/18 路文件身份、顺序、RGB、像素 K、物理外参／射线、浮点 mask 和相对深度；不能只比 shape。SVF-GS 内部采用 OpenGL 相机轴，本项目采用 OpenCV 轴，比较时须配对转换射线与外参。
- 加载器不访问禁止的资产；测试前向不依赖 Metric3D／mask／DA V2；修改目标 GT 或 bin 顺序不能改变生成高斯。
- 112×200／224×400 都渲染回完整原图；边缘主点／射线投影正确，补边像素不参与对齐、不生成高斯。
- 阶段 1 只有尺度模块有梯度；阶段 2/3 的允许模块得到梯度；π³、LPIPS、天空头保持冻结。全无效深度和空体素情况不静默丢弃评估 bin。
- 动态 mask 下 RGB 与 LPIPS 的计算匹配 SVF-GS 规则；全图评估不沿用 loss mask。
- 合成及真实小样本核对渲染 accumulated Z、PCC 组内展平和跨 bin 汇总；检测遗漏／重复 token、非有限值和不匹配的输出元数据。
- 在实际 GPU 上完成两个分辨率的前向／反向／渲染 smoke check，之后才能宣称 CUDA、显存或训练链路可用。

最初文档轮次仅制定方案；后续已按用户授权完成适配与有界检查，具体记录见第 11 节。SVF-GS、DepthSplat 和现有 Conda 环境未被修改；未启动正式训练、未实际发送飞书消息、未提交或推送 Git。

## 10. 审阅时需要保留的结论

本次产物应标识为 **“UniSplat 在单帧 OmniScene 协议下的适配结果”**。它保留原默认基础模型、空间体素结构、双高斯分支和三阶段训练思想，但按已确认边界取消时序记忆／天空标签／动态 BCE，并用 Metric3D 替代 LiDAR 监督。不能把后续结果直接称为论文原始 nuScenes 设置的复现。

影响模型定义和预算的选择已经逐项询问并得到确认。用户随后授权实施以及下载基础权重；权重位置、检查结果与正式实验边界见第 11 节。

## 11. 实施记录与运行方法

### 11.1 已落地的代码

| 功能 | 实现位置 |
| --- | --- |
| 两个分辨率入口 | `configs/experiment/omniscene_112x200.yaml`、`omniscene_224x400.yaml` |
| 配置组合、协议校验、全局步数调度 | `omniscene/config.py` |
| 现有资产加载、独立 context／target／supervision | `dataset/omniscene.py` |
| 冻结 π³、三阶段参数设置、原尺寸渲染 | `omniscene/model.py` |
| 无历史静态 head、Metric3D 对齐、排除补边 | `model/gaussian_head/static_head.py` |
| 射线与精确体素特征查询 | `omniscene/geometry.py` |
| 原损失权重与 SVF-GS 浮点掩码 | `omniscene/losses.py` |
| 优化器更新预算、恢复与验证／mini／final 事件 | `omniscene/training.py`、`checkpoint.py` |
| 全图四指标、分组统计、计时与完整性检查 | `omniscene/evaluate.py`、`metrics.py` |
| 有界异步通知、持久待发队列、恢复去重 | `omniscene/notify.py` |
| 训练／独立完整评估 | `train_omniscene.py`、`eval_omniscene.py` |

运行时不 import SVF-GS 或 DepthSplat。只有独立的 `tools/check_omniscene_protocol.py` 验收脚本读取本地参考项目函数。原 Waymo 入口保持原调用方式；共享 head 的新增构造选项具有原默认值，U-Net 在 `history_infos=None` 时绕过历史融合，静态 head 不持有历史队列。

实现时明确了以下兼容性细节：

- **DINOv2 位置编码**：官方 ViT-S/14 reg4 权重为 37×37 网格（518 预训练尺寸），两种实验均以 `dinov2_pretrain_img_size=518` 构造参数，再由原 DINO 插值到当前 patch 网格。严格加载全部权重，不丢弃位置编码。这与原 Waymo 518 宽构造相符，也使两个分辨率参数量一致。
- **相机坐标**：SVF-GS 对 c2w 的 Y/Z 轴翻转，同时其相机射线 Y/Z 也翻转；UniSplat／DepthSplat 使用原始 OpenCV c2w。因此应比较转换后的物理射线，而非直接要求两套 c2w 数字相同。真实样本已核对等价。
- **体素查询**：原 `simple_knn_v2` 最近邻结果只有量化中心距离为 0 才有效。静态路径使用整数体素键实现同一 occupied-cell 查询，不依赖本环境未安装的 `simple_knn_v2`；不改体素尺寸或空间范围。
- **显存**：`Loss.checkpoint_perceptual=true`、`Model.checkpoint_rendering=true` 分别重算冻结 VGG 的激活及每个目标视角的渲染缓冲。全部 18 路监督、原损失和梯度保留；没有降低分辨率、减少目标或改变 batch。推理计时不执行这些训练重算分支。
- **异常深度**：全部六路 Metric3D 对齐均无效时明确报错；单相机无效时，阶段 1 仅统计有效相机，阶段 2 对该相机回退到已冻结的尺度预测，日志记录有效相机数。阶段 3／推理不读取 Metric3D。
- **恢复完整性**：模型和训练状态分别保存 SHA-256；mini 结果身份绑定 checkpoint／配置／清单。事件日志也绑定 checkpoint 哈希。最终 mini 被中断时，恢复同一步评估，不再增加训练更新。

### 11.2 权重与依赖

用户授权后，原始 [π³ 权重](https://huggingface.co/yyfz233/Pi3/tree/ae722e7039287d0c8fde9f11f197f804f44b510c) 和 [DINOv2 ViT-S/14 reg4 权重](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth) 已下载到本仓库 `ckpt/`，不是 UniSplat 的 Waymo 训练权重。

| 权重 | 默认本地相对路径 | SHA-256 |
| --- | --- | --- |
| 原始 π³ | `ckpt/pi3/model.safetensors` | `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d` |
| DINOv2 ViT-S/14 reg4 | `ckpt/dinov2/dinov2_vits14_reg4_pretrain.pth` | `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb` |
| VGG16 | `ckpt/lpips/vgg16-397923af.pth` | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| LPIPS 系数 | `ckpt/lpips/vgg.pth` | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` |

VGG／LPIPS 使用本机已有官方缓存复制到项目目录。来源 URL、文件大小与校验值保存在 `ckpt/sources.json`；`tools/download_omniscene_weights.py` 固定本次版本和哈希，文件齐全时只校验，不联网。训练与评估不会临时下载权重。

已在现有 `unisplat` 环境（Python 3.10、PyTorch 2.6.0+cu118、RTX 4090）验证，无环境安装或修改。当前 RoPE 使用上游自带 PyTorch fallback；计时输出记录其 backend，不能将它当成 CUDA RoPE 性能。π³ 按本协议使用 BF16，需要支持 BF16 的 GPU。

### 11.3 启动、恢复和评估

以下命令在 UniSplat 根目录运行。默认飞书模块根目录为 `/home/dzp62442/Libraries`，实际模块为其下的 `auto_monitor/send_feishu/` 包；webhook 沿用该模块已有配置，不写入本项目。正式训练按已授权规则自动推送训练开始和 mini 评估日志。通知失败有最多三次尝试和每次 15 秒超时，保存在运行目录 `notifications/`，训练不会因通知网络错误中断。

```bash
conda activate unisplat
export PYTHONNOUSERSITE=1

# 可单独运行配置／资产检查，不构建网络，不发通知。
python train_omniscene.py --config configs/experiment/omniscene_112x200.yaml --check-config
python train_omniscene.py --config configs/experiment/omniscene_224x400.yaml --check-data

# 两个独立实验；同一张 GPU 上分别启动，不建议同时运行。
python train_omniscene.py --config configs/experiment/omniscene_112x200.yaml
python train_omniscene.py --config configs/experiment/omniscene_224x400.yaml

# 恢复小分辨率实验：重新执行原命令，自动找到最新 checkpoint。
python train_omniscene.py --config configs/experiment/omniscene_112x200.yaml

# 可选：手动指定恢复点，优先于自动查找。
python train_omniscene.py --config configs/experiment/omniscene_112x200.yaml \
  --resume work_dirs/unisplat_omniscene_static_112x200/checkpoints/step_050000

# 正式完整测试：30080 bins；大分辨率替换实验名及配置即可。
python eval_omniscene.py --config configs/experiment/omniscene_112x200.yaml \
  --checkpoint work_dirs/unisplat_omniscene_static_112x200/checkpoints/step_100001 \
  --split total
```

默认训练输出为 `work_dirs/<实验名>/`；独立评估输出为 `outputs/<实验名>/step_<步数>_<checkpoint哈希前缀>/<mini或total>/`。可用 `--work_dir` 或 `--output-dir` 指定独立目录。CLI 支持 dotlist，例如 `Dataset.num_workers=4`、`Feishu.enabled=false`；更改已锁定的 6/18 视角、时序／天空／LiDAR 边界、阶段预算等会直接报错。

自动续训只扫描本次实验目录的 `checkpoints/step_<步数>/`，按数值步数选择最新正式保存点，不跨分辨率或其他实验目录搜索；使用自定义 `--work_dir` 后，再次传入相同目录即可自动恢复。保存中的 `.tmp` 目录不参与选择，`latest` 软链接缺失、失效或滞后时仍按正式目录判断。恢复前沿用模型／训练状态哈希、配置身份、数据清单以及事件日志配对检查；最新正式保存点损坏时明确报错，不自动回退或覆盖。此前未保存的优化器更新会从最近 checkpoint 重跑。

最终步已经达到预算但 mini 未完成时，只恢复并完成最终评估，不增加更新次数。已有 `training_complete.json` 时，还会核对最终 mini 的配置／checkpoint／数据身份、三组逐 bin 指标、完整性及计时记录；确认完整后在构建模型与初始化 CUDA 之前退出。结果缺失时继续评估。需要重新训练时使用新的运行目录；手动从更早 checkpoint 分叉时，也应指定新的 `--work_dir`，避免覆盖已有的后续 checkpoint 和评估。

训练输出包括解析后的配置、源码版本／dirty 标记、权重和数据清单身份、本地训练日志、各阶段参数统计、配对 checkpoint、验证结果和 mini 指标。mini 固定发生在 50000、60000、70000、80000、90000、100000、100001；每 1000 步的 10-bin 验证仍在三个阶段执行。

### 11.4 验证证据与边界

CPU 验收命令为 `python -m unittest discover -s tests -v`，当前 21 项检查全部通过。用合成资产及小模型覆盖配置、精确调度、阶段冻结、浮点 mask 平方权重、禁用资产读取、空／缺失／重复／非有限指标、体素查询梯度、RNG／优化器恢复和通知失败重试。故障注入覆盖阶段 1/2 边界、阶段 2 中途及最终 mini 中断后的自动恢复，恢复后参数与不中断的 CPU 小模型训练逐值一致；另验证 latest 丢失／滞后／失效、临时保存目录、已完成运行跳过、结果缺失后补评估和显式恢复优先级。

`python tools/check_omniscene_protocol.py` 在两个分辨率各抽查两个真实 bin，逐值检查 RGB、DA V2 相对深度、像素 K、相机射线和浮点 loss mask；报告位于 `work_dirs/smoke/protocol.json`。这是样本协议核对，不代表逐文件审计全部 135932／30080 个 bin。

GPU 检查脚本：

```bash
python tools/smoke_omniscene.py --config configs/experiment/omniscene_112x200.yaml \
  --output work_dirs/smoke/112x200_final.json --metrics --optimizer-step
python tools/smoke_omniscene.py --config configs/experiment/omniscene_224x400.yaml \
  --output work_dirs/smoke/224x400_final.json --metrics --optimizer-step
```

脚本只对真实单样本执行每阶段一次前向／反向（可选一次仅在内存中的优化器更新）、两个 bin 的评估流程及合成高斯渲染检查；不保存训练权重，不启动 100001 步实验，不发送飞书消息。检查了冻结参数、有限损失／梯度、完整尺寸、跨 bin 无历史状态、渲染 `accumulated_z`、偏置主点及重算前后的渲染梯度一致性。其分数与耗时属于未训练模型的冒烟记录，**不能作为 mini／完整测试对比结果或正式重建耗时报告**。

两个分辨率实测模型参数量相同（不含 loss／metric 网络）：

| 阶段 | 可训练 | 冻结 | 总参数 |
| --- | ---: | ---: | ---: |
| 1 | 70,329,346 | 1,036,512,184 | 1,106,841,530 |
| 2／3 | 77,729,421 | 1,029,112,109 | 1,106,841,530 |

包含已注册但禁用的天空参数及 π³ 未使用输出头，并在明细中标注；两个分辨率的点锚定高斯分别恰为 134400／537600，补边不额外生成点锚。224×400 单样本阶段 2 检查中，启用重算后的 PyTorch 峰值 allocated 约 13.0 GiB；这不是全数据集显存上限或进程总显存承诺。

尚未运行任一分辨率的正式长程训练、2048-bin mini 或 30080-bin 完整评估。因此尚无训练收敛、正式 PSNR／SSIM／LPIPS／PCC 或正式完整重建耗时结论。后续正式实验直接使用上述入口，产物仍应标识为本协议下的静态 UniSplat 适配结果。
