# MedCTA 数据与四个本地工具准备结果

2026年10月26日，**准备完成，四个本地工具全部通过合成输入功能验收**。107张发布图像全部可解码。GoogleSearch所需Serper key按用户要求留待以后配置；没有启动训练、正式医学基准评分或RadA后续工作。

## 验收结果

| 项目 | 实际检查 | 结果 |
|---|---|---|
| 发布图像 | 仅从Parquet读取image与image_path，107张逐张解码并保存 | 107/107通过 |
| Calculator | 计算 `2 + 3 * 4` | 返回14 |
| OCR | 合成图片含MEDICAL 123，要求识别出123 | 通过，结果29字符 |
| ImageDescription | 对同一合成图片实际生成描述 | 非空输出253字符 |
| RegionAttributeDescription | 对指定区域实际生成属性描述 | 非空输出255字符 |

两个视觉工具共享同一个已加载的LLaVA推理器，实际执行了两次生成，不只检查导入或文件存在。最终READY记录和退出码0同时存在，准备控制进程已结束。描述返回非空属于功能验收，不证明描述准确性、医学有效性或持续学习收益。没有用107条发布基准的问题、答案或轨迹调参，也没有同集训练后报告泛化结果。

## 实际新增存储与时钟

| 计量范围 | 实际分配字节 |
|---|---:|
| 数据、代码、权重及本次缓存 | 14,400,614,400 |
| 独立工具环境的新增部分 | 289,497,088 |
| 临时目录 | 4,096 |
| **最终新增合计** | **14,690,115,584（14.690 GB）** |
| **守护进程记录的新增峰值** | **14,693,707,776（14.694 GB）** |

GB使用十进制10^9字节。统计覆盖三个本次新建目录，基线均为0；已有PyTorch环境只读复用，其既有占用不重复计入。存储检查每2秒执行，45,000,000,000字节预停，严格上限50,000,000,000字节，临时分段、依赖安装、缓存和所有准备重试均计入。

首次实际启动为05:53:30 UTC，成功退出为08:08:17 UTC，总历时8086.84秒，约2小时14分47秒。所有修复和续跑共用原始09:53:30 UTC截止时间，没有重置四小时时钟。

## 下载、版本与兼容修复

数据来自[MedCTA固定发布版本](https://huggingface.co/datasets/IVUL-KAUST/MedCTA/tree/0be777092121a18ba2a85a0c4145a9d4fe8ed7db)，工具定义来自[固定源码提交](https://github.com/IVUL-KAUST/MedCTA/blob/eb7d1dc0adc6da4c31d9c5ebef3a8059a620022c/agentlego/benchmark.py)。此发布包取得时不需要MIMIC权限；这不代替对每张图像原始来源的审计。

两个视觉工具采用同一份13,518,899,358字节LLaVA主权重和608,502,127字节CLIP权重。OCR使用EasyOCR的英文检测/识别模型。HF仅取得必要tokenizer/config，没有额外下载完整Llama或第二份视觉模型。

短时测速为：代理单连接1.91MB/s、模型源直连单连接3.41MB/s、四路直连合计8.89MB/s。随后采用四路64MiB分段并复用已有前缀，最多256MiB临时分段，逐段核对HTTP206、Content-Range及长度再顺序追加。8.89MB/s是短时测速，不能解释为全程恒定速度。直连只用于download.openmmlab.com并限制同主机重定向；GitHub操作继续经代理。

在独立工具环境内修复了旧软件与现有硬件的实际兼容问题：

- Torch2.8与MMEngine0.10.4重复注册Adafactor，显式保留原Transformers优化器注册；共享PyTorch环境未修改。
- 源码压缩包有两处文档logo符号链接，解包时跳过这两处文档链接，其他链接仍拒绝。
- 补齐tifffile与pycocotools；后者是MMPretrain启用真实多模态模型注册的必要依赖。
- LLaVA旧包装器与Transformers自动SDPA选择不兼容，显式采用eager attention，并修复MMPretrain适配器向from_config转发此选项。没有修改模型结构或权重。
- 验收前实时检查显存，原卡余量不足时选择另一张满足至少38,000MiB空闲余量的GPU，不修改其他任务。

最终运行使用Torch2.8.0+cu128、Torchvision0.23.0+cu128、Transformers4.40.1、MMEngine0.10.4、MMCV-lite2.1.0、MMPretrain1.2.0、EasyOCR1.7.1。失败和受控重启记录保留，没有把失败尝试从预算中扣除。

## 公开材料与复用边界

[脱敏验收与资源摘要](../docs/evidence/medcta_preparation_summary_20261026.json)给出数值、实际解析的HF revision及验收边界。[准备脚本](../scripts/prepare_medcta.py)、[独立守护进程](../scripts/medcta_preparation_supervisor.py)、[功能验收脚本](../scripts/check_medcta_local_tools.py)和[部署配置模板](../configs/medcta_preparation.template.json)一并公开。

模板中的TASK_ROOT、TOOL_ENV、TMPDIR与首次/截止时间必须由部署者填写；本次现有Torch基础环境的只读复用也需在新机器上提供。模板固定了本次实际记录的两项HF tokenizer/config revision；原运行先解析再记录版本。输入Parquet应预先保存为TASK_ROOT/input/train.parquet，脚本放在TASK_ROOT/src。HOME、HF_HOME/HF_HUB_CACHE、TORCH_HOME、MMENGINE_HOME、EASYOCR_MODULE_PATH、XDG_CACHE_HOME及MPLCONFIGDIR等缓存目录均需限定在计量根内。PRIVATE_ARGV_TERM用于设置本机私有命令行禁用词；公开守护脚本以此环境变量替代原私有用户名文字。所有计算入口仍通过环境变量和中性runpy入口启动。

医学图像、QA数据、模型权重、机器私有路径、进程身份记录、凭据及原始日志没有公开。当前就绪范围是发布数据和四个本地工具；缺少Serper配置，尚未验证完整五工具代理工作流，也没有正式MedCTA分数。
