# Case study: ChaosNLI, and what a rank gap is worth

Three audits, on real data, all reproducible from this directory.

**Finding 1 — three boards, three verdicts, two refusals.** On the three
ChaosNLI components the same gate returns `READABLE` once and `UNREADABLE`
twice. On both refusals the *leader itself* sits at or below the noise floor of
the labels it was measured against — no model on those boards beats what one
human annotator reproduces from the other 99.

**Finding 2 — a floor is a property of who produced the labels, not of the
board.** The same nine checkpoints, the same 1,000 items, the same margin. Swap
the floor's producer from 100 humans to 32 named LLM judges and the verdict
inverts: `READABLE` becomes `UNREADABLE`, because an LLM panel agrees with
itself (0.8660) far more than any model in the board achieves (best 0.7890).

**Finding 3 — the judges move further from re-ordering the answer options than
the board's top ranks move from swapping one model for another.** The same
judge, on the same 500 items, scored against the human majority, swings by a
median of **0.0120** when nothing changes but the order the options are
presented in. The board's top-to-second gap on those same items is **0.0020**.
The ordering artefact is **6x the gap** it would be used to compute, and it
exceeds that gap for **all 32 judges**, not just the worst few. Order is still
the *smaller* of the two artefacts — a judge reproduces itself across option
orders (median 0.9317) better than the panel reproduces itself across judges
(0.8616) — which is why finding 2, not finding 3, is the one that moves the
floor.

This is `arx`'s central claim — *a leaderboard's rank difference is readable
only if it exceeds the noise floor of the labels it was measured against* — run
on data instead of on a worked example.

## Audit 1: the three components

| component | task | n | floor | leader | leader acc | margin | MDE | verdict |
|---|---|---:|---:|---|---:|---:|---:|---|
| `snli` | 3-way | 1,514 | 0.7484 | roberta-large | 0.7867 | **+0.0383** | 0.0203 | `READABLE` |
| `mnli_m` | 3-way | 1,599 | 0.6416 | roberta-large | 0.6354 | **−0.0062** | 0.0255 | `UNREADABLE` |
| `alphanli` | 2-way | 1,532 | 0.8786 | albert-xxlarge | 0.8473 | **−0.0313** | 0.0238 | `UNREADABLE` |

The two refusals are refusals of the form arx is built to produce. `mnli_m`'s
leader is 0.62 points *below* the floor; `alphanli`'s is 3.13 points below it.
The gate prints the leader's own accuracy next to the floor it failed to clear,
so the reader can see the number rather than a hedge.

And even where the gate says `READABLE`, it says nothing about the ranking
underneath. The top-to-second gaps against the resolution bound:

| component | top1–top2 | as × MDE | top1–top4 | as × MDE |
|---|---:|---:|---:|---:|
| `snli` | 0.0040 | **0.19x** | 0.0059 | **0.29x** |
| `mnli_m` | 0.0169 | 0.66x | 0.0432 | 1.69x |
| `alphanli` | 0.0104 | 0.44x | 0.0339 | 1.43x |

On `snli` — the one board that is readable — the top four systems are within
**0.29x** of the smallest difference the apparatus can detect. `READABLE` is a
statement about the leader against the floor. Four systems a leaderboard renders
as ordered ranks are, by this measurement, indistinguishable.

## Audit 2: hold the board fixed, swap the floor

Same nine checkpoints, same 1,000-item subsample of ChaosNLI-S, same
`per_item_scores`, same declared delta. Only the label producers change:

| floor source | floor | leader | leader acc | margin | MDE | verdict |
|---|---:|---|---:|---:|---:|---|
| 100 human annotations (expanded) | 0.7467 | albert-xxlarge | 0.7890 | **+0.0423** | 0.0321 | `READABLE` |
| 32 named LLM judges (real identities) | 0.8660 | albert-xxlarge | 0.7890 | **−0.0770** | 0.0321 | `UNREADABLE` |

Both floors satisfy every premise the gate checks: the producers are disjoint
from the contestants (humans or judges, versus checkpoints), and both are
computed on the board's own item set.

The panel release's own numbers, for context:

* LOO agreement **among the 32 judges** is 0.8660 — quoted here as a diagnostic,
  because the panel cannot be both the judge set and the contestant set. The
  gate refuses that, correctly.
* The best individual judge (`gem31pro`) reaches 0.8530 against the human
  majority — **still below the panel's collective 0.8660**. Median 0.8205,
  worst 0.7120 (`dsv32`).

So the ranking is not readable *or* unreadable in the abstract. It is readable
against human-produced labels and unreadable against model-produced labels, on
identical evidence. Anyone who reports "the leaderboard says model X is best"
without naming which floor the claim rests on has left out the half of the
sentence that decides it.

One more piece of evidence for the same point, from audit 1: the leader on the
1,000-item subsample is `albert-xxlarge`, while on all 1,514 items it is
`roberta-large`. The top of this board does not survive a change of sample
size either.

## Audit 3: the same judge, three option orders

The board's top ranks are separated by a fraction of the resolution bound, and
those gaps come from labels, and the labels came through a prompt. So: how far
does one judge move on one item when nothing about the item changes and only
the order in which the answer options are presented is permuted?

The panel release ships that arm — 500 of the 1,000 items, re-asked under three
permutations, all 32 judges — and its own analysis guide lists
presentation-order analyses under **"Not covered"**, so this segment fills a gap
the release names rather than re-deriving a published number.

| on the 500 swap items | value |
|---|---:|
| board top1–top2 gap | 0.0020 |
| board top1–top4 spread | 0.0060 |
| **order-induced accuracy swing**, same judge, same items | median **0.0120**, p75 0.0200, max 0.0340 |
| label flip rate vs variant 0 | median 7.1%, max 15.8% |
| **swing / top1–top2 gap** | **6.00x** |

A flip rate and a score change are not the same thing — flips cancel — so the
headline is the *accuracy swing*, scored against the human majority under each
permutation, not the flip rate. The conclusion holds either way, and it holds
for every judge: **all 32 of them swing by more than the board's top gap.**

The layering matters, and it goes against the obvious reading. Option order is a
*smaller* perturbation than judge identity:

| artefact | magnitude |
|---|---:|
| same judge across option orders (self-reproducibility, LOO over 3 permutations) | median **0.9317** |
| different judges, same item (panel LOO floor) | **0.8616** |

So a judge reproduces itself across formatting *better* than the panel
reproduces itself across judges. Both artefacts dwarf the top gap, but only the
second one moves the floor — which is why finding 2 is the load-bearing one.

Two things this is not:

* **Not a repeated-call noise floor.** The release states plainly that this
  archive establishes no same-prompt repeated-call floor. An order flip is an
  artefact of presentation, and the two are not interchangeable.
* **Not a verdict.** The three permutations are one judge, not three independent
  producers, so a floor over them would measure that judge's self-consistency
  across formatting. It is reported as a diagnostic above, never as a gate
  result.

One more check, because the release excludes one judge (`dsv32`) from its own
panel analysis so that a single panel is used across datasets. Both readings
hold, so neither headline finding depends on that choice:

| panel | LOO floor | floor = humans | floor = panel |
|---|---:|---|---|
| all 32 judges | 0.8616 | `MARGINAL` | `UNREADABLE` |
| 31 judges (release protocol) | 0.8693 | `MARGINAL` | `UNREADABLE` |

Note the human-floor verdict on *these* 500 items is `MARGINAL`, where the same
board over all 1,000 items was `READABLE`. The effect did not move; the
resolution did. That is what having four distinct verdicts is for.

The release also publishes a SHA-256 per vote file; all 64 (32 baseline, 32
swap) are verified against what is on disk by this script. That is stronger than
the first-seen recording that predated it.

## How the human floor is built, and the one modelling step

The public ChaosNLI release publishes each item's vote **counts** — a 2- or
3-vector summing to 100 — and **not** worker identities, so a
`(generators x items)` matrix cannot be read off the file. The audit script
expands each item's counts into 100 exchangeable pseudo-annotators with a
seeded shuffle.

That is a derived quantity, not a measured one, and it is labelled as such.
Every assignment consistent with the released counts is equally likely given
those counts, so the leave-one-out floor it produces is a draw of "how often
one annotator reproduces the majority of the other 99". Repeating the expansion
under three seeds moves the floor by **at most 2.2e-16** — the construction is
stable rather than assumed to be.

The judge floor needs no such step: the panel release carries one vote per item
per judge, with the judge named.

## Sources, with hashes

Both scripts download at run time, verify SHA-256, and keep the raw files out of
the repository. ChaosNLI is CC-BY-NC-4.0 and is not redistributed here.

| what | where | SHA-256 |
|---|---|---|
| ChaosNLI-S annotations | HF mirror `earino/chaosnli`, `raw/chaosNLI_snli.jsonl` | `99f9015d…` |
| ChaosNLI-M annotations | same mirror, `raw/chaosNLI_mnli_m.jsonl` | `8eb49b58…` |
| ChaosNLI-α annotations | same mirror, `raw/chaosNLI_alphanli.jsonl` | `f55817ec…` |
| 9-model per-item predictions | vendored copy of the upstream release, `TheGuy-26/chaosnli-deberta-modernbert` | `7c788bb9…` |
| αNLI per-item predictions | same vendored copy | `58ced70a…` |
| 32-judge panel votes (baseline + swap arms) | `Chao1208/32judges-votes`, `datasets/chaosnli-snli-1000/` | SHA-256 published per file in the release's `manifest.json`; all 64 files re-verified on disk by `audit_position_bias.py` |
| panel item list (1,000 uids) | same release, `items/uids.txt` | `f0e5bea8…` (matches the release's own manifest) |

Three caveats, stated plainly rather than buried:

* **The model predictions are a mirror, not upstream.** The upstream
  `scripts/download_data.sh` points at a Dropbox zip; that URL answered with an
  HTML interstitial when this was run. The annotations' hashes match the values
  recorded in the mirror's own audit file, and both scripts recompute them
  rather than trusting that file.
* **The judge panel's backend is not independently authenticated.** That is the
  release's own caveat, quoted from `panel-chaosnli.json`: the model strings are
  what was requested, not necessarily what served the request. It bounds what
  finding 2 can mean — the panel's *agreement level* is measured here, its
  *identity* is taken on trust.
* **These are 2019-era checkpoints, not LLMs.** The subject of audit 1 is the
  apparatus — 100 human labels with a measurable floor. The gate is indifferent
  to which models are ranked, which is why audit 2 could hold the board fixed
  and move only the floor.

## Reproduce

```bash
python3 audit_chaosnli.py                     # audit 1: all three components
python3 audit_judge_panel.py                  # audit 2: the two-floor comparison
python3 audit_position_bias.py                # audit 3: the option-order arm
python3 audit_chaosnli.py --offline           # reuse whatever is in _data/
```

Committed output: [`results.json`](results.json) (audit 1),
[`results_judge_panel.json`](results_judge_panel.json) (audit 2), and
[`results_position_bias.json`](results_position_bias.json) (audit 3). `_data/` is
git-ignored. The write-up aimed at readers outside this repository is in
[`ARTICLE.md`](ARTICLE.md) / [`ARTICLE.zh-CN.md`](ARTICLE.zh-CN.md).

## What these do not establish

* They do not say which checkpoint is the better model. They say whether this
  measurement could resolve the difference, and on two of three boards it could
  not.
* They do not say the human majority is ground truth. ChaosNLI exists precisely
  because it is not: 14 items in `snli` have no majority at all, and the mean
  majority share is 0.755.
* They do not transfer to a board that publishes only aggregate scores. A floor
  needs per-item labels from independent producers; without them the apparatus
  cannot be checked at all.

### Why not AlpacaEval

The obvious target was a current LLM leaderboard. The AlpacaEval 2.0 data was
downloaded and checked, and it fails the gate's premises:

* its per-item **human** labels are not published — only an aggregate `Human
  agreement` column (e.g. `alpaca_eval_gpt4_fn` = 70.99%);
* its five public evaluators are four prompt variants of the same GPT-4 plus one
  Claude run covering a single model, so a leave-one-out floor over them would
  measure GPT-4's self-consistency, not an inter-annotator noise floor;
* two of those variants agree on **805 of 805** items — one judge run, not two
  producers;
* the ranked contestant set includes `gpt4` and `gpt-3.5-turbo-0301` while the
  judges are GPT-4-family, an actor-disjointness violation.

The denominators were fine (the three evaluators cover exactly the same 805
items), so the failure is not a mismatched item set — it is that the labels have
no independent producers to build a floor from. That is itself the check this
gate exists to perform.

---

# 中文部分

**结论一：三个组件，三个判决，两次拒绝。** 同一道门在 ChaosNLI 三个组件上给出 1 次 `READABLE`、2 次 `UNREADABLE`；两次拒绝中，**榜首本身就在它被测的那个标签噪声地板之下**——那些 board 上没有任何模型比「一个人类标注者复现其余 99 人的多数意见」更强。

| 组件 | 任务 | n | 地板 | 榜首 | 榜首准确率 | margin | MDE | verdict |
|---|---|---:|---:|---|---:|---:|---:|---|
| `snli` | 3 类 | 1,514 | 0.7484 | roberta-large | 0.7867 | **+0.0383** | 0.0203 | `READABLE` |
| `mnli_m` | 3 类 | 1,599 | 0.6416 | roberta-large | 0.6354 | **−0.0062** | 0.0255 | `UNREADABLE` |
| `alphanli` | 2 类 | 1,532 | 0.8786 | albert-xxlarge | 0.8473 | **−0.0313** | 0.0238 | `UNREADABLE` |

而且即便判 `READABLE`，它对**下层的排名什么都没说**：`snli` 上第一名到第四名合计只差 **0.0059，仅为 MDE 的 0.29 倍**。四个被榜单渲染成「有序名次」的系统，在这次测量下无法区分。

**结论二：地板是「谁产生的标签」的属性，不是 board 的属性。** 固定同一张 9 模型 board、同一批 1,000 个 item、同一个 margin，只把地板的产生者从 100 个人类换成 32 个具名 LLM 判官，判决直接翻转——`READABLE` 变 `UNREADABLE`。原因是 LLM 面板自我一致率（0.8660）远高于 board 里任何模型的成绩（最好 0.7890）。

| floor 来源 | floor | 榜首 | 准确率 | margin | MDE | verdict |
|---|---:|---|---:|---:|---:|---|
| 100 个人类标注（展开） | 0.7467 | albert-xxlarge | 0.7890 | **+0.0423** | 0.0321 | `READABLE` |
| 32 个具名 LLM 判官（真身份） | 0.8660 | albert-xxlarge | 0.7890 | **−0.0770** | 0.0321 | `UNREADABLE` |

补充诊断：**32 个判官内部**的 LOO 一致率 = 0.8660（只作诊断——把面板同时当作判官集与参赛者集属于 actor-disjointness 违反，门会拒绝，这是正确的）；最好的单个判官 `gem31pro` 对人类多数票达 0.8530，**仍低于面板集体的 0.8660**；中位 0.8205，最差 `dsv32` 0.7120。

所以这张榜单不是「抽象地可读或不可读」：在人类产生的标签下可读，在模型产生的标签下不可读，证据完全相同。任何只报「榜单说 X 模型最好」而不说明这个判断建立在哪个地板上的人，都漏掉了决定它的那半句话。

另一个同向证据：在 1,000 子样本上榜首是 `albert-xxlarge`，而在全部 1,514 例上是 `roberta-large`——这张榜单的顶端**连样本量变化都扛不住**。

**结论三：判官因「调换选项顺序」而移动的幅度，大于榜首与第二名之间被用来排名的差距。** 同一判官、同一批 500 个 item、对人类多数票计分，**只改选项呈现顺序**，准确率中位波动 **0.0120**；而这批 item 上 board 的 top1–top2 差距是 **0.0020**——序效应是它将要参与计算的差距的 **6 倍**，且**全部 32 个判官**（不只是最差的几个）都超过该差距。

分层很重要，而且与直觉相反：**选项序是比「换判官」更温和的扰动**——同一判官跨三种选项序的自我复现率中位 **0.9317**，高于面板跨判官的一致率 0.8616。两者都远大于 top1–top2 差距，但只有后者会移动地板，这就是为什么结论二才是承重的那条。

这一段**不是重复发布方的结论**：其分析指引明确把 presentation-order 分析列在 **"Not covered"** 之下，本段是补它标出的空白。也正因如此必须写清两点：① 它**不是**重复调用噪声地板——发布方明确声明该档案**不建立**同 prompt 重复调用的地板，序翻转是**呈现格式的伪影**，两者不可互换；② 它**不是 verdict**——三种排列是同一个判官，不是三个独立生产者，因此据此算出的地板只反映该判官跨格式的自我一致性，本报告只作诊断呈现。

顺带做了发布方协议的敏感性检验（它出于「跨数据集共用同一面板」的规则把 `dsv32` 排除在自己的面板分析之外）：32 判官下 loo = 0.8616、31 判官下 = 0.8693，**两种口径下「人类地板 → 可读/边缘、面板地板 → 不可读」的翻转都成立**，两条主结论都不依赖这个选择。同时注意：**这 500 个 item 上人类地板的判决是 `MARGINAL`，而同一 board 在全部 1,000 个 item 上是 `READABLE`**——效应没动，分辨率动了，这正是四种判决分立的用途。

**地板的构造（唯一一处建模步骤，已明示）**：ChaosNLI 公开发布的是每例**投票计数**（2 或 3 维、和为 100），不含标注者身份。脚本按每个 item 的计数做**种子化随机展开**成 100 个可交换的伪标注者；这是**派生量而非实测量**，三个种子下地板变动不超过 **2.2e-16**。判官地板**不需要**这一步——面板发布的是每个判官每 item 一条具名投票。

**来源与哈希**：ChaosNLI 三个组件标注分别为 `99f9015d…` / `8eb49b58…` / `f55817ec…`（HF 镜像 `earino/chaosnli`）；9 模型与 αNLI 逐项预测分别为 `7c788bb9…` / `58ced70a…`（`TheGuy-26` 里 vendored 的上游副本）；判官投票来自 `Chao1208/32judges-votes`（其 `manifest.json` **逐文件公布 SHA-256**；32 baseline + 32 swap 共 64 个文件已全部在磁盘上复核通过）；面板 item 列表哈希 `f0e5bea8…` 与其 manifest 记载一致。原始数据一律不入库（ChaosNLI 为 CC-BY-NC-4.0），脚本运行时下载并校验。

**三点保留**：① 模型预测来自**第三方镜像**，上游 Dropbox 链接在本次执行时返回 HTML 拦截页；标注哈希与镜像自带审计文件记载一致，且脚本是重新校验而非采信。② **判官面板的后端未经独立认证**——这是发布方自己的 caveat（引自 `panel-chaosnli.json`）：模型字符串是「请求的」而非「应答的」。这限定了结论二的含义：这里测的是面板的**一致水平**，其**身份**是采信的。③ 这 9 个模型是 **2019 年代的编码器，不是 LLM**；审计对象是**装置**，这正是结论二能「固定 board、只换地板」的原因。
