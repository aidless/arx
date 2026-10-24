# Short launch copy (draft) — arx

> Status: draft. Not published. Facts to keep in sync: 370 passed, 3 skipped;
> Apache-2.0; no PyPI package (install from checkout or GitHub); Python 3.12+.

## Show HN

**Title:** Show HN: arx – ask whether a leaderboard can resolve a comparison,
before asking if the effect is big

**Body:**

Most eval debates ask whether a difference is big enough. arx asks whether the
measurement can resolve the comparison at all.

It returns four verdicts — READABLE / MARGINAL / UNDECIDABLE / UNREADABLE — and
a fifth check that runs first: is the noise floor above chance? Five generators
who each flip a coin agree half the time; any score above that looks like a
landslide. If the floor is indistinguishable from chance, the verdict is
UNDECIDABLE regardless of the margin.

The demo that convinced me: the same board gives MDE 0.0274 (independent
binomial) vs 0.0743 (paired cluster bootstrap) — a 2.7× difference, and the
verdict flips from READABLE to MARGINAL with no change to the data. That is why
`mde_method` is always in the report.

Six examples run offline, no API key, no model:
`git clone https://github.com/aidless/arx && cd arx && pip install -e . &&
python3 examples/02_audit_a_leaderboard.py`

Apache-2.0, Python 3.12. It is not a general-purpose agent and does not claim a
research result. Counter-examples welcome.

## r/MachineLearning

**Title:** arx: an evaluation gate that refuses claims the measurement cannot
resolve

**Body:**

I kept seeing leaderboard differences reported without a resolution bound. arx
makes that bound a first-class output: floor, margin, MDE, and the method that
produced the MDE.

Four verdicts, not one: READABLE / MARGINAL / UNDECIDABLE / UNREADABLE. The
distinction matters because the fixes differ — more items vs. a different design
vs. fixing the measurement.

One concrete result from the examples: the same board gives MDE 0.0274 under an
independent binomial approximation and 0.0743 under a paired cluster bootstrap,
flipping the verdict. The fallback was optimistic because it assumed away the
item-level correlation.

The repository also has a loop that generates hypotheses and runs experiments,
but the gate is usable on its own and has no model dependency. Six offline
examples, 370 passing tests, Apache-2.0, Python 3.12.

No PyPI package yet — `git clone` + `pip install -e .`. The live demo renders
eight audit cases with the reason each refusal fired. I am more interested in
where the verdicts are wrong than in stars.

## X / Mastodon thread (4 posts)

1. A leaderboard difference is not a measurement until you know what it can
   resolve. I built arx to make that bound the first thing a report prints:
   floor, margin, MDE, and the method behind the MDE.
2. Four verdicts, not one: READABLE / MARGINAL / UNDECIDABLE / UNREADABLE.
   "Smaller than we can tell" and "the question cannot be answered as asked" are
   different failures with different fixes.
3. Same board: MDE 0.0274 (binomial approx) vs 0.0743 (paired bootstrap) — the
   verdict flips with no change to the data. That is why mde_method is always in
   the report.
4. Six offline examples, no API key, Python 3.12, Apache-2.0. Not a
   general-purpose agent; the gate is the point. Tell me where it is wrong.

## 中文社区（知乎 / 即刻 / V2EX）

**标题：** 把“这个榜单差异能不能被分辨”做成工具：arx

**正文：**

大多数评测讨论在问“这个差异够不够大”。arx 先问前一步：**这份测量有没能力
分辨这个比较。**

它给出四种判决，而不是一个“显著/不显著”：
READABLE（可读）/ MARGINAL（小于可分辨量）/ UNDECIDABLE（按当前设计根本答不了）/
UNREADABLE（测量本身坏了）。前两种的修法不同：一个要加样本，一个要改设计。

先跑的那道检查是：噪声地板是否高于随机水平。五个只会抛硬币的生成器彼此一致率
约 50%，地板就贴着 0.5，任何高于它的分数都像大胜。地板分不清随机，就没有“金标准”
可测，判决一律 UNDECIDABLE。

最能说明问题的一个例子：同一份榜单，独立二项近似给出 MDE 0.0274，配对
cluster bootstrap 给出 0.0743，判决从 READABLE 翻成 MARGINAL，数据一个字没变。
所以报告里永远打印 `mde_method`。

六个例子离线可跑，不需要 API key：
`git clone https://github.com/aidless/arx && cd arx && pip install -e . &&
python3 examples/02_audit_a_leaderboard.py`

Apache-2.0，Python 3.12。它不是通用 agent，也不声称某个研究结论成立。
比起 star，我更想要“这个判决判错了”的反例。

## Posting notes

- Do not claim a research result. The repo itself says the RL demo is a bandit.
- Do not imply a PyPI package exists until one does.
- If someone shows a board where the gate is wrong, reproduce it and either fix
  the gate or add it to the known-limits section before replying.
- Post the long version first; wait a day before the short ones.
