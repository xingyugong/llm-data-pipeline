# LLM 数据工程 + 微调项目 · 讲透版 / Deep-Dive Tutorial

**学完需要多久 / Estimated time: 45–60 分钟**（读 25 分钟 + 自己跑一遍 pipeline 20 分钟 + 背面试回答 10 分钟）
**作者 / Author:** Xingyu Gong · **语言 / Language:** 中文讲解 + 英文关键词（面试时直接用英文词）

---

## 0. 一句话说清这个项目 / The project in one sentence

> 中文：我写了一个 Python 程序，把一份真实的人类指令数据集（Dolly 15k）清洗干净、去掉重复和隐私信息、切成不泄漏的训练集/测试集，然后用 LoRA 方法微调一个小语言模型，并用测试集评估它有没有变好。
>
> English: I built a Python pipeline that cleans a real human-written instruction dataset (Dolly 15k), removes duplicates and PII-like patterns, splits it without leakage, then fine-tunes a small language model with LoRA and evaluates it on a held-out test set.

**它不是什么 / What it is NOT:** 它不是我自己发明了一个新模型，也不是用 AI 生成了假数据。模型是现成的 distilgpt2，数据是真实的 Dolly，我做的核心工作是**数据工程 + 训练流程 + 评估**。面试官最看重的恰恰是这部分。

---

## 1. 关键词先行 / Keywords first（先背词，再看概念）

| 英文术语 | 中文 | 一句话理解 |
|---|---|---|
| Instruction tuning | 指令微调 | 教模型“按指令回答”，不是教新知识 |
| LoRA (Low-Rank Adaptation) | 低秩适配 | 只训练一小部分新增参数，冻结大模型本体 |
| Base model | 基座模型 | 预训练好的大模型，这里是 distilgpt2（82M 参数） |
| Fine-tune | 微调 | 在基座模型上继续训练，让它适应某个任务 |
| Dedup / De-duplication | 去重 | 删掉重复的例子，防止模型死记硬背 |
| Decontamination | 去污染 | 保证测试集的例子没在训练集里出现过 |
| Data leakage | 数据泄漏 | 测试集偷偷混进了训练数据，评估分数是假的 |
| PII (Personally Identifiable Information) | 个人身份信息 | 邮箱、电话等，训练数据里必须去掉 |
| Perplexity | 困惑度 | 越低越好；模型对正确答案“有多意外”，越不意外越好 |
| Held-out test set | 留出测试集 | 训练时完全没见过的数据，用来诚实评估 |
| Shingle / Jaccard similarity | 词组重叠度 | 判断两段文本“长得有多像”的方法 |
| Hash split | 哈希切分 | 用哈希值决定例子进训练集还是测试集，确定、可复现 |
| Overfitting | 过拟合 | 模型背下了训练集，换新数据就不行了 |
| Adapter | 适配器 | LoRA 训练出来的小权重文件，插到基座模型上就能用 |

**面试金句（背下来）:**
> "The model code is the short part. Data engineering — dedup, quality filtering, PII removal, and leakage-free splits — is most of the work."
> （模型代码是短的。数据工程——去重、质量过滤、去隐私、无泄漏切分——才是大部分工作。）

---

## 2. 为什么要做这个？/ Why this project?

### 2.1 背景：大模型是怎么来的（三步）

1. **Pre-training 预训练**：模型读海量文本，学会“下一个词最可能是什么”。得到 base model（基座模型）。它会说话，但不听指令。
2. **Instruction tuning 指令微调**：给模型看成千上万个“指令 → 好回答”的例子，教它按指令格式回答。这就是本项目做的事。
3. **RLHF / 偏好对齐**（本项目不做）：用人类打分进一步调教。这是下一步，不是现在。

### 2.2 为什么数据比模型重要？

一个常见误解：“微调 = 换个更大的模型”。错。

- 垃圾数据进 → 垃圾模型出（Garbage in, garbage out）。如果训练集里有重复 100 遍的例子，模型会把那句话背下来，而不是学会能力。
- 如果测试集泄漏进训练集，你的评估分数是假的，面试官一问“你怎么切分的？”就露馅。
- 真实工业界（Google、Meta、OpenAI 的数据团队）的大部分时间都花在数据 pipeline 上，而不是调模型结构。**这就是为什么这个项目对口 Data Engineering for LLM。**

### 2.3 为什么选 Dolly，不自己编数据？

- Dolly 15k 是 Databricks 发布的、**人类写**的 15,011 条指令/回答，CC BY-SA 3.0 许可证（可商用、可公开，注明来源即可）。
- 自己用 AI 生成数据叫 synthetic data（合成数据），容易有幻觉、质量不可控，而且面试官会问“你怎么验证生成质量？”——很难回答。
- 用真实数据 + 公开可复现的清洗流程，任何人都能下载同一份数据、跑同一段代码、得到同样的 1,937 条。这是**可复现性 (reproducibility)**，是工程能力的证明。

---

## 3. 整体结构 / Architecture

```
                ┌─────────────────────────────┐
                │  Dolly 15k (Hugging Face)   │
                │  15,011 条真实指令/回答       │
                └──────────────┬──────────────┘
                               │ load_dataset()
                               ▼
                ┌─────────────────────────────┐
                │  src/clean_data.py          │
                │  ① 精确去重 ② 质量过滤       │
                │  ③ 近似去重 ④ 哈希切分       │
                └──────────────┬──────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        data/processed/               data/processed/
        train.jsonl (1,829)           test.jsonl (108)
                │                             │
                ▼                             ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ src/train_lora.py│      │ src/evaluate.py  │
        │ LoRA 微调        │      │ 困惑度 + 生成对比 │
        │ distilgpt2       │      │ base vs tuned    │
        └──────────────────┘      └──────────────────┘
                │
                ▼
        outputs/lora-distilgpt2/  (只含 adapter 小文件，不含 82M 基座模型)
```

**三个文件各干什么（记住这个分工，面试必问）:**

| 文件 | 输入 | 输出 | 一句话 |
|---|---|---|---|
| `clean_data.py` | 原始 Dolly | 干净的 train/test JSONL + 质量报告 | 数据工程 |
| `train_lora.py` | train.jsonl | LoRA adapter 权重 | 训练 |
| `evaluate.py` | adapter + test.jsonl | 困惑度对比 + 生成样本 | 评估 |

---

## 4. 数据清洗，一步一步讲 / Data cleaning, step by step

**真实运行结果（2026-10-28，limit=2000）：** 2,000 条原始 → **1,937 条保留**（1,829 训练 / 108 测试）。被拒：44 条回答太短、8 条太长、3 条含 URL、2 条含疑似隐私、6 条近似重复。

### Step 1 — 精确去重 Exact dedup

**做什么 / What:** 把 instruction + context + response 全部小写、压缩空白后算 SHA-256 哈希，哈希相同 = 完全相同的例子，只留第一条。

**为什么 / Why:** 同一条数据出现多次，模型训练时会反复看到它，等于变相加权。更糟的是：同一条可能同时出现在训练集和测试集 → 泄漏。

**代码核心 / Core code:**
```python
def example_key(instruction, context, response):
    return hashlib.sha256("\x00".join(
        [normalize(instruction), normalize(context), normalize(response)]
    ).encode("utf-8")).hexdigest()
```
`normalize` = 小写 + 把多个空白压成一个空格。`"Hello  World"` 和 `"hello world"` 会被认成同一条——这正是我们想要的。

**如果去掉这步会怎样？** 训练集里藏着测试集的原题，困惑度虚低，面试官问切分方法时无法自圆其说。

### Step 2 — 质量过滤 Quality filters

**做什么 / What:** 每条例子过 6 条规则，任何一条不过就丢掉，并记下原因（`quality_ok` 返回原因字符串，`None` = 通过）：

| 规则 | 阈值 | 为什么 |
|---|---|---|
| 指令太短 | < 5 字符 | 不是有效指令 |
| 回答太短 | < 10 字符 | 没信息量，模型学不到东西 |
| 回答太长 | > 4,000 字符 | 可能是粘贴的整篇文章，训练时占太多 token |
| 非英文过重 | ASCII 占比 < 90% | Dolly 是英文数据集，混入其他语言会污染 |
| 疑似隐私 | 出现邮箱/电话正则 | PII 绝不能进训练数据（合规红线） |
| 回答含 URL | 有 `http://` | 很多是广告/引用，回答本身不完整 |
| 重复字符 | 单个字符占 > 50% | 如 `aaaaaaa...`，是损坏数据 |

**为什么用正则找 PII？** 正则表达式 (regular expression) 是模式匹配：`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` 意思是“字母数字 + @ + 域名”。它不完美（会漏掉变形写法），但简单、透明、可解释。工业界会在此基础上再加更强的检测器——面试时可以这样说，显得你知道边界。

### Step 3 — 近似去重 Near-duplicate removal

**做什么 / What:** 两条例子不是一字不差，但意思/措辞几乎一样（比如只换了两个词），也要只留一条。

**怎么做 / How — shingle + Jaccard（重点，面试常考）:**
1. **Shingle（词组切片）:** 把文本切成连续 5 个词的窗口。例如 `"the cat sat on the mat"` → `{"the cat sat on the", "cat sat on the mat"}`。
2. **Jaccard 相似度:** `J(A,B) = |A∩B| / |A∪B|`，即两个集合交集大小 ÷ 并集大小。完全一样 = 1.0，完全不同 = 0.0。
3. 阈值 **≥ 0.85** 判定为近似重复，丢掉后出现的。

```python
def shingles(text, k=5):
    words = normalize(text).split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)
```

**为什么不用“意思相近”的语义方法（embedding）？** 语义方法需要再跑一个模型，又贵又慢，而且阈值难解释。Shingle/Jaccard 是工业界去重的标准轻量做法（Google 的网页去重就用类似思想），15k 这个量级完全够用。**局限要诚实说：** 它是 O(n²)  pairwise 比较，数据量到百万级要用 MinHash/LSH（局部敏感哈希）加速——README 里已经写了这一点。

### Step 4 — 哈希切分 Hash split（防泄漏的关键）

**做什么 / What:** 每个例子算一个 split key = `SHA-256(规范化后的 instruction + context)`，取前 8 位十六进制转成 0–1 的数，小于 0.05 → 测试集，否则 → 训练集。

**为什么不用 `random.shuffle` 随机切？**
- 随机切分有两个问题：① 每次跑结果不一样，**不可复现**；② 如果同一指令的两个变体分别进了训练集和测试集，随机切分发现不了。
- 哈希切分是**确定性**的：同一条数据永远进同一个 split；而且 split key 只用 instruction+context，不用 response，所以**同一个 split key 永远不可能同时出现在两边**——泄漏在结构上就不可能发生。

**为什么测试集只有 5%？** 微调评估不需要大测试集，108 条足够算困惑度；训练数据越多越好。5% 是常见取舍。

### Step 5 — 格式化 Format

每条变成统一文本：
```
### Instruction:
<指令>

### Input:
<上下文，可为空>

### Response:
<回答>
```
**为什么固定格式？** 模型在训练时学会“看到 `### Response:` 后面就是要生成的回答”，推理时给它同样的格式，它就知道该干什么。这叫 **prompt format consistency（提示格式一致性）**——训练格式和推理格式必须一样，否则效果打折。

---

## 5. LoRA 微调 / LoRA fine-tuning（核心概念，最难也最重要）

### 5.1 先理解：全量微调为什么贵？

distilgpt2 有 82M 参数。全量微调 = 更新全部 82M 个数，需要存模型 + 梯度 + 优化器状态，显存轻松翻 3–4 倍。对 7B 模型，全量微调普通人根本跑不起。

### 5.2 LoRA 的想法：别动大矩阵，加个小补丁

**核心公式 / Core formula:**
```
W' = W + B·A
```
- `W`：基座模型里原来的权重矩阵（比如 768×768），**冻结，一动不动**。
- `B·A`：两个很瘦的小矩阵相乘。`B` 是 768×8，`A` 是 8×768。`8` 就是 **rank r**。
- 训练时只更新 `B` 和 `A`，`W` 保持原样。

**算一笔账（面试加分题）：**
- 原矩阵参数：768 × 768 = 589,824
- LoRA 参数：768×8 + 8×768 = 12,288
- **只训练原来 2% 的参数。** r 越小越省，但表达能力越弱；r=8 是小模型的常用起点。

**为什么低秩（low-rank）够用？** 经验发现：微调时权重真正需要的“变化量”其实藏在一个低维空间里——大模型已经会语言了，微调只是把它往“按指令回答”这个方向推一把，不需要把整个矩阵重写。这就是 LoRA 论文的核心洞察。

**为什么只对 `c_attn` 和 `c_proj` 用 LoRA？**
GPT-2 的每个 Transformer 层里，`c_attn` 负责算注意力（决定“看哪些词”），`c_proj` 负责把注意力结果投影回去。注意力机制是模型“理解指令结构”的地方，微调这里性价比最高。只动这两处，进一步省参数。

### 5.3 训练流程（`train_lora.py` 逐段）

1. **读数据**：`load_rows` 读 train.jsonl 的 `text` 字段，上限 `--max-examples`。
2. **分词**：`tokenizer` 把文本切成 token ID（模型只懂数字）。`max_length=512` 截断过长的。
3. **冻结 + 挂 LoRA**：`get_peft_model` 把 `W` 冻结，给 `c_attn/c_proj` 挂上 `B·A`。
4. **Trainer 训练**：学习率 2e-4（LoRA 常用，比全量微调高，因为只训小矩阵）、batch 4、2 个 epoch。`bf16` 只在有 GPU 时开。
5. **只保存 adapter**：`trainer.save_model` 存的是 `B·A` 小文件（几 MB），不是整个模型。用的时候：加载 distilgpt2 + 插上 adapter = 微调后的模型。

**Smoke test（冒烟测试）：** `--max-examples 20 --max-steps 5`，只训 5 步，目的是**证明代码能跑通**，不是为了效果。工业界术语：smoke test = “先确认不冒烟（不报错），再谈效果”。

### 5.4 训练还没跑——诚实声明

这台 VM 没有 GPU，训练脚本**只做了语法检查（py_compile），没有真正跑完**。简历和 README 里都如实写了这一点。下一步：在 Colab T4（免费 GPU）上跑 `--max-examples 2000 --epochs 2`，再跑评估，拿到真实数字后再往简历上加结果。**永远不要在没跑过的情况下写“困惑度下降了 X%”。**

---

## 6. 评估 / Evaluation（怎么知道变好了？）

### 6.1 困惑度 Perplexity

**定义：** 对测试集里的正确答案，模型平均每个 token 的“意外程度”取指数。公式：
```
Perplexity = exp(平均每个 token 的 loss)
```
- Loss 越小 → 模型越“料到”正确答案 → 困惑度越低 → 越好。
- **必须在 held-out 测试集上算**，在训练集上算没有意义（背下来了当然低）。

**对比逻辑：** 同一个测试集，分别算 base 模型和 LoRA 模型的困惑度。如果 tuned < base，说明微调确实让模型更会按指令回答了——这就是 `evaluate.py` 做的事。

### 6.2 生成对比 Generation

困惑度是数字，生成是肉眼检查：给 3 个固定 prompt，让 tuned 模型贪心生成（greedy decoding，每步选概率最高的词，不随机），把输出原样存下来。面试官能直接读到“模型回答得像不像人话”。

**为什么用 greedy 而不用 sampling？** Greedy 是确定性的，同样输入永远同样输出，**可复现**；sampling 每次结果不同，没法公平对比。

---

## 7. 怎么跑 / How to run

```bash
# 1. 数据 pipeline（本机 CPU 就能跑，已跑通过）
pip install -r requirements.txt
python -m src.clean_data --limit 2000 --out-dir data/processed \
    --report reports/data_quality_report.md

# 2. 测试（5 个，验证去重/过滤/切分逻辑）
pytest tests -q

# 3. 训练（Colab T4 免费 GPU，见 README）
pip install torch transformers peft accelerate datasets
python -m src.train_lora --data data/processed/train.jsonl \
    --output-dir outputs/lora-distilgpt2 --max-examples 2000 --epochs 2

# 4. 评估（训练跑完之后）
python -m src.evaluate --adapter outputs/lora-distilgpt2 \
    --test data/processed/test.jsonl --out reports/evaluation.md
```

**全量版：** 把 `--limit 2000` 改成 `--limit 15011`（整个 Dolly），重新跑 pipeline 再训练。

---

## 8. 局限与下一步 / Limitations & next steps（面试时主动说，显得诚实）

1. **模型小且老**：distilgpt2 是 2020 年的 82M 模型，结论不能直接推广到 7B+ 现代模型；但 pipeline 和 LoRA 机制是通用的。
2. **数据通用，不针对经济**：Dolly 是通用指令数据。下一步可以换一个有许可证的经济/金融 QA 数据源，同一套清洗代码复用。
3. **近似去重是 O(n²)**：15k 没问题，百万级要上 MinHash/LSH。
4. **PII 正则不完美**：只能抓标准格式的邮箱/电话，变形写法会漏网，生产环境要加更强的检测器。
5. **还没跑训练**：所有训练/评估数字都要等 Colab 跑完再填。

---

## 9. 面试问答 / Interview Q&A（中英对照，背粗体部分）

**Q1: What did you build? 你做了什么？**
> A bilingual answer: I built a data pipeline for LLM instruction tuning. It cleans Dolly 15k — dedup, quality and PII filtering, leakage-free split — and LoRA fine-tunes distilgpt2, evaluated on a held-out set.
> 我做了一个 LLM 指令微调的数据 pipeline：清洗 Dolly 15k（去重、质量和隐私过滤、无泄漏切分），用 LoRA 微调 distilgpt2，在留出集上评估。

**Q2: Why not just use the raw dataset? 为什么不直接用原始数据？**
> Raw data has duplicates, PII-like patterns, and leakage risk. Duplicates overweight examples; leakage makes evaluation meaningless. Cleaning is most of the work.
> 原始数据有重复、隐私、泄漏风险。重复会加权，泄漏让评估作废。清洗才是大部分工作。

**Q3: How did you prevent leakage? 怎么防泄漏？**
> Hash split on normalized instruction+context. The same split key can never appear in both train and test, and the split is deterministic and reproducible.
> 对规范化后的指令+上下文做哈希切分，同一个 key 永远不可能同时出现在两边，且结果确定、可复现。

**Q4: Explain LoRA in one minute. 一分钟讲 LoRA。**
> Freeze the base weights W, train only a low-rank update BA with rank 8. For a 768×768 matrix that's 12k trainable parameters instead of 590k — about 2%. The update lives in a low-dimensional subspace, which is enough to steer the model toward following instructions.
> 冻结基座权重 W，只训练低秩更新 BA（rank=8）。768×768 的矩阵从 59 万参数降到 1.2 万，约 2%。更新量藏在低维空间里，足够把模型推向“按指令回答”。

**Q5: How do you know it worked? 怎么知道有效？**
> Held-out perplexity, base vs tuned, on the same test split — plus greedy generations I can read. I don't report numbers I haven't run.
> 同一留出集上对比 base 和 tuned 的困惑度，再加可读的贪心生成样本。没跑过的数字我不报。

**Q6: What would you do with 10x more data? 数据量大 10 倍怎么办？**
> Replace pairwise near-dup with MinHash/LSH, parallelize cleaning, and move training to multi-GPU with gradient checkpointing. The pipeline interface stays the same.
> 近似去重换 MinHash/LSH，清洗并行化，训练上多卡 + 梯度检查点。接口不变。

---

## 10. 自测 / Self-check（答不出就回看对应节）

1. Shingle Jaccard = 0.9 的两条数据，留还是删？为什么？（§4 Step 3）
2. 为什么 split key 不用 response？（§4 Step 4）
3. r=8 时，768×768 矩阵的 LoRA 参数是多少？（§5.2）
4. 困惑度是在训练集还是测试集上算？为什么？（§6.1）
5. 如果面试官问“训练结果怎么样？”，你现在应该怎么回答？（§5.4：还没跑，诚实说）
