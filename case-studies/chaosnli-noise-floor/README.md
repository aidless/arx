# Case study: ChaosNLI-S, and what a rank gap is worth

**Finding.** The top model on ChaosNLI-S is genuinely above the noise floor of
its own labels. The ranking *below* it is not: the top four systems sit within
0.6 percentage points of each other, against a resolution bound of 2.03
percentage points. The leader is readable; the leaderboard around it is not.

This is `arx`'s central claim run on real data instead of on a worked example.

| | |
|---|---|
| dataset | ChaosNLI-S (1,514 items, 100 human annotations each) |
| contestants | 9 transformer checkpoints, per-item predictions published |
| gold floor | leave-one-out agreement of 100 expanded annotators = **0.7484** |
| chance level | 0.3333 |
| leader | `roberta-large` = **0.7867** |
| margin over floor | **0.0383** |
| MDE (paired, cluster bootstrap) | **0.0203** |
| **verdict** | **`READABLE`** — no refusals |

## Two readings, and only one of them is the verdict

The gate returns `READABLE`. Read that carefully, because it is a narrow claim:

1. **The leader clears the floor.** `roberta-large` is 3.83 points above the
   level a single human annotator reproduces from the other 99. The smallest
   difference this apparatus could detect is 2.03 points. 0.0383 > 0.0203, so
   the leader's position is not an artefact of the label noise.
2. **The ranking inside that board is not resolvable.** First to second is
   **0.0040** — **0.2x the MDE**. First to fourth is **0.0059** — **0.3x**.
   Four systems that a leaderboard renders as ordered ranks are, by this
   measurement, indistinguishable.

`READABLE` is a statement about the *leader against the floor*. It says nothing
about places two through four, and the numbers above are what that gap looks
like. A board that prints "1. roberta-large, 2. bart-large" without this
bound is printing a ranking the data cannot support.

## The board

Accuracy against the majority human label, all 1,514 items, identical item set
for every system and for the floor:

| rank | system | accuracy |
|---:|---|---:|
| 1 | roberta-large | 0.7867 |
| 2 | bart-large | 0.7827 |
| 3 | albert-xxlarge | 0.7814 |
| 4 | xlnet-large | 0.7807 |
| 5 | roberta-base | 0.7536 |
| 6 | bert-large | 0.7384 |
| 7 | xlnet-base | 0.7365 |
| 8 | bert-base | 0.7292 |
| 9 | distilbert | 0.7021 |

## How the floor is built, and the one modelling step

The stage is set by ChaosNLI itself: 100 human annotations per item, so the
labels have a measurable, published noise floor rather than an assumed one.

The public release publishes each item's vote **counts** — a 3-vector summing
to 100 — and **not** worker identities. A `(generators x items)` matrix
therefore cannot be read off the file, so the audit script expands each item's
counts into 100 exchangeable pseudo-annotators with a seeded shuffle.

That is a derived quantity, not a measured one, and it is labelled as such
here. Every assignment consistent with the released counts is equally likely
given those counts, so the leave-one-out floor it produces is a draw of "how
often one annotator reproduces the majority of the other 99". Repeating the
expansion under three seeds moves the floor by **1.1e-16** — the construction
is stable rather than assumed to be.

Two further things the gate checked and did not refuse:

* **actor-disjointness** — the floor is produced by human annotators, the board
  by models. The two sets are disjoint.
* **common denominator** — every one of the nine systems has predictions on
  all 1,514 items, and the floor is computed on that same set. `n_floor_items`
  is not a subset of the board's items; it is the board's item set.

## Sources, with hashes

The script downloads at run time, verifies SHA-256, and keeps the raw files out
of the repository — ChaosNLI is CC-BY-NC-4.0 and is not redistributed here.

| what | where | SHA-256 |
|---|---|---|
| annotations | HF mirror `earino/chaosnli`, `raw/chaosNLI_snli.jsonl` | `99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1` |
| model predictions | vendored copy of the upstream release in `TheGuy-26/chaosnli-deberta-modernbert` | `7c788bb9df6917f56a99a3bbf19c307bf2d02ad14d8f02537338b7bfb629bb53` |

Two caveats worth stating plainly rather than burying:

* **The model predictions are a mirror, not upstream.** The upstream
  `scripts/download_data.sh` points at a Dropbox zip; that URL answered with an
  HTML interstitial when this was run, so the copy above is a third-party
  vendored one. The annotations' hash matches the value recorded in the
  mirror's own audit file, and the script re-checks it rather than trusting it.
* **These are 2019-era checkpoints, not LLMs.** BERT/XLNet/RoBERTa-family
  encoders. The subject of the audit is the *apparatus* — 100 human labels with
  a measurable floor — and the gate is indifferent to which models are ranked.
  The same script runs unchanged on an LLM board that publishes per-item
  labels from multiple independent producers. See the note below on why no
  current LLM leaderboard was used instead.

## Reproduce

```bash
python3 audit_chaosnli.py --config snli      # downloads, verifies, audits
python3 audit_chaosnli.py --config mnli_m    # the companion 1,599-item board
python3 audit_chaosnli.py --offline          # reuse whatever is in _data/
```

Committed output is in [`results.json`](results.json); it is what the run above
prints. `--seeds` controls how many times the count expansion is repeated.

## What this does not establish

* It does not say `roberta-large` is a better model. It says this measurement
  can resolve its position over the floor, and it does not affect the models
  below it.
* It does not say the human majority is ground truth. ChaosNLI exists precisely
  because it is not: 14 items here have no majority at all, and the mean
  majority share is 0.755.
* It does not transfer to a board that publishes only aggregate scores. A floor
  needs per-item labels from independent producers; without them the
  apparatus cannot be checked at all.

### Why not AlpacaEval

The obvious target for this audit was a current LLM leaderboard. The AlpacaEval
2.0 data was downloaded and checked, and it does not satisfy the gate's
premises:

* its per-item **human** labels are not published — only an aggregate `Human
  agreement` column (e.g. `alpaca_eval_gpt4_fn` = 70.99%);
* its five public evaluators are four prompt variants of the same GPT-4 plus
  one Claude run that covers a single model, so a leave-one-out floor over them
  would measure GPT-4's self-consistency, not an inter-annotator noise floor;
* two of those variants (`alpaca_eval_cot_gpt4_turbo_fn`,
  `weighted_alpaca_eval_gpt4_turbo`) agree on **805 of 805** items — they are
  one judge run, not two producers;
* the ranked contestant set includes `gpt4` and `gpt-3.5-turbo-0301` while the
  judges are GPT-4-family, which is an actor-disjointness violation.

The denominators were fine — the three evaluators cover exactly the same 805
items — so the failure is not a mismatched item set. It is that the labels have
no independent producers to build a floor from. Recording that is itself a
result: it is the check this gate exists to perform, and it is why the audit
below rests on a dataset that publishes 100 independent human annotations per
item.

---

# 中文部分

**结论**：ChaosNLI-S 榜首位置是可信的，**但榜首以下的排名不可信**。

| | |
|---|---|
| 数据集 | ChaosNLI-S（1,514 例，每例 100 个人工标注） |
| 参赛者 | 9 个 transformer 检查点，逐项预测公开 |
| 标签噪声地板 | 100 个展开标注者的 leave-one-out 一致率 = **0.7484** |
| 随机水平 | 0.3333 |
| 榜首 | `roberta-large` = **0.7867** |
| 高出地板 | **0.0383** |
| MDE（配对，cluster bootstrap） | **0.0203** |
| **verdict** | **`READABLE`**，无任何拒绝理由 |

**必须分开读的两条结论**：arx 判 `READABLE`，它是一个很窄的断言——① 榜首的 0.0383 高出地板、且大于该装置的最小可检测差 0.0203，所以榜首位置不是标签噪声造成的；② **但前四名的差距只有 0.0059，仅为 MDE 的 0.3 倍**（第一名 vs 第二名仅 0.0040，0.2 倍）。四个被榜单渲染成「有序名次」的系统，在这次测量下**无法区分**。

`READABLE` 是关于「榜首与地板」的断言，它对第 2～4 名什么都没说——而上面的数字就是那个「没说」的样子。一个只印「1. roberta-large、2. bart-large」却不给这个分辨率下界的榜单，印的是一个数据支撑不了的排名。

**地板的构造（唯一一处建模步骤，已明示）**：ChaosNLI 公开发布的是每例的**投票计数**（3 维向量、和为 100），**不含标注者身份**，所以无法直接读出 `(生成器 × item)` 矩阵。脚本按每个 item 的计数做**种子化随机展开**成 100 个可交换的伪标注者。这是**派生量、不是实测量**，本报告据实标注：与已发布计数一致的任何指派在给定计数下等概率，因此得到的 LOO 地板是「一个标注者复现其余 99 人多数意见的频率」的一次抽样。三个种子下地板变动 **1.1e-16**，构造是稳定的，不是假定的。

**来源与哈希**（脚本运行时下载并校验 SHA-256；ChaosNLI 为 CC-BY-NC-4.0，本仓不重分发原始数据）：

- 标注：HF 镜像 `earino/chaosnli` 的 `raw/chaosNLI_snli.jsonl`，SHA-256 = `99f9015d…`
- 模型预测：`TheGuy-26/chaosnli-deberta-modernbert` 里 vendored 的上游发布副本，SHA-256 = `7c788bb9…`

**两点需明说的保留**：① 模型预测来自**第三方镜像**——上游 `download_data.sh` 指向的 Dropbox 链接在本次执行时返回 HTML 拦截页；标注的哈希与镜像自带审计文件记载一致，且脚本是重新校验而非采信。② 这 9 个模型是 **2019 年代的编码器，不是 LLM**；本审计的对象是**装置**（100 个人工标注及其可测地板），与所排名的模型无关，同一脚本可原样跑在「发布多独立生产者逐项标签」的 LLM 榜单上。

**不成立的推论**：本审计不说 `roberta-large` 是更好的模型；不说人类多数票是标准答案（14 例根本无多数，平均多数占比 0.755）；也不适用于只发布聚合分数的榜单——地板需要独立生产者的逐项标签，缺了它装置根本无法被检验。

**为什么不用 AlpacaEval**：曾下载核对过。它的**逐项人类标签未发布**（只有聚合的 `Human agreement = 70.99%`）；5 个公开判官是同一 GPT-4 的 4 个提示变体 + 1 个仅覆盖单模型的 Claude 运行，据此算 LOO 地板只会测出 GPT-4 的自我一致性，而非标注者间噪声；其中两个变体在 **805/805** 个 item 上判定完全相同（同一次判官运行）；且被排名的 `gpt4`、`gpt-3.5-turbo-0301` 与 GPT-4 家族判官构成 actor-disjointness 违反。分母本身没问题（三个判官恰好覆盖同一批 805 项），失败在于**标签没有独立生产者**可供构造地板。把这个记下来本身就是一个结果。
