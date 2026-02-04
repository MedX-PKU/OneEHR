下面是我建议你明天的 TODO（按“重要/阻塞程度”排序），都是我们目前**还没完全实现或还欠完善**的点：

**1) DL 模型持久化 & `oneehr test` 支持 DL**
- 现状：`oneehr test` 现在只支持 tabular（`xgboost/catboost/rf/dt/gbdt`）。DL（`gru/rnn/lstm/transformer/...`）训练后虽然会写 `metrics.json`/`preds`，但没有统一的“可复现加载 + 推理评估”路径。
- TODO：
  - 统一保存 DL 模型：`state_dict.pt` + `model_meta.json`（包含 `model.name`、超参、input_dim、static_dim、prediction_mode、feature_columns/code_vocab 版本等）
  - `oneehr test` 增加 DL 推理/评估（patient/time 两种）

**2) Tabular 模型在训练阶段的持久化一致性检查**
- 现状：已支持保存/加载，但训练阶段目前只确保保存 `feature_columns.json` 与模型文件；缺少“保存 postprocess pipeline”与“推理时复用同一 pipeline”的强一致。
- TODO：
  - 将 `oneehr/data/postprocess.py` 的 fitted pipeline 以机器可读格式持久化（例如 joblib 或更严格的 JSON schema）
  - `oneehr test` 加载并应用训练时的 postprocess（否则 train/test 数据分布可能不一致）

**3) Static feature 的工程化（目前是 MVP 级别）**
- 现状：static 从同一事件表抽取 `cols`，聚合 first/last，并在 `align_static_features` 里对非数值做 category codes（很粗）。
- TODO：
  - static feature 走与 tabular 同样的 postprocess（one-hot、标准化、缺失值策略）
  - static 与 dynamic 的 feature name/version 记录到 artifacts，保证可复现

**4) 部分模型目前是“正确但慢”的 baseline（N-N）**
- 现状：`retain` / `grasp` 的 time 模式用了 prefix 循环（O(T²)），数据一大就会慢。
- TODO：
  - 优化这些模型的 N-N 实现（矢量化或缓存）
  - 或者在配置里明确标注 “time mode not recommended” 并给出 warning

**5) 模型实现对齐 PyEHR 原版行为（严格复刻）**
- 现状：我们为了“干净 + 接入链路”，对 `dragent/concare/grasp` 等做了结构化重写；接口已统一，但不保证与 PyEHR 原实现逐行等价。
- TODO：
  - 逐个模型做 “对齐测试”：在同一随机输入下对比 PyEHR 输出（容许数值误差）
  - 需要的话把关键模型改成更贴近原论文/原代码的实现

**6) 文档/示例补齐**
- 现状：模型/配置项已经很多，但缺少按模型分类的 TOML 示例。
- TODO：
  - 在 `examples/` 增加多个 config：tabular、sequence patient、sequence time、static+dynamic
  - 在 `docs/` 写一页 “model.name 列表 + 每个模型支持的 prediction_mode + 必要配置段”
