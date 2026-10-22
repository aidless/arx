# arx 榜单可读性审计 Demo

**一句话**：arx 不回答“这个效应够不够大”，它先回答“这份测量有没有能力分辨这个比较”。
八个合成案例在构建时由 `arx.audit.audit_leaderboard` 重算，报告由
`arx.audit.report_html` 生成；页面只负责展示，不重新实现判决逻辑。

## 30 秒开始

```bash
cd arx
python3 demo/server.py                  # http://127.0.0.1:8000/
```

本地模式会先跑八个审计，然后提供：

- `GET /api/health`
- `GET /api/cases` / `GET /cases.json`
- `POST /api/run`：重新执行八个案例
- `GET /config.json`：`{"mode": "live"}`

静态回放：

```bash
python3 demo/build_static.py            # 写 demo/public/cases.json
python3 -m http.server -d demo/public   # 任意静态服务器
```

构建脚本会核对每个案例的预期判决；任何一个偏差都会以非零退出码结束。

## 测试

```bash
python3 -m unittest demo/test_demo.py -v
```

覆盖：八个案例的判决、敏感性网格、ChaosNLI 真实案例、Report HTML 的边界、
已提交 `cases.json` 与当前代码一致、UI 不硬编码指标、无密钥/绝对路径、
本地 HTTP API、以及“扫描器能红”的反证夹具。

## 八个案例

| id | 看点 | 预期判决 |
|---|---|---|
| `readable` | 五名生成者，领先者高出噪声地板 | `READABLE` |
| `marginal` | 同样的榜面，只有 120 个 item | `MARGINAL` |
| `unreadable` | strong=0.86 没清过标签自身的复现率 | `UNREADABLE` |
| `two_generators` | 去掉一个评审者后地板无法估计 | `UNDECIDABLE` |
| `actors_overlap` | 评审者同时是参赛者 | `UNDECIDABLE` |
| `noise` | 五名生成者只在掷硬币，地板接近随机 | `UNDECIDABLE` |
| `domain_mismatch` | 地板用 400 个 item，榜面只用 150 个 | `UNDECIDABLE` |
| `single_generator` | 只有一名 gold 生成者，无法做 leave-one-out | `UNDECIDABLE` |

## 两个加深视图

- **分辨率敏感性**：同一个 400-item pool 的嵌套前缀，n 从 40 到 400，声明 delta 从
  0.005 到 0.05；每个格子是 `audit_leaderboard(n_boot=500)` 的判决。小样本是
  `MARGINAL`，大样本在较小 delta 下变 `READABLE`，而 delta=0.05 会把它推回 `MARGINAL`。
- **真实数据案例**：ChaosNLI 同一榜面，human floor → `READABLE`，32 名 LLM judge floor
  → `UNREADABLE`。页面回放 `case-studies/chaosnli-noise-floor/results_judge_panel.json`
  的审计结果，不重算语料。

## 三分钟面试讲法

1. 先讲问题：榜单会显示排名，但排名差是否大于标签噪声，是另一个问题。
2. 点 `readable`：地板 0.9210，领先者 0.9600，margin +0.0390 > 分辨率下界 0.0274。
3. 点 `marginal`：同样的榜面只有 120 个 item，margin 落在分辨率内；结论从“可读”变成“再采数据”。
4. 点 `unreadable`：领先者 0.8600 低于地板 0.9210；榜单上的名次没有测量支撑。
5. 点 `two_generators` / `actors_overlap` / `noise` / `domain_mismatch` / `single_generator`：
   五种 `UNDECIDABLE` 不是失败，是拒绝给一个没有依据的数。
6. 看敏感性网格：小样本 `MARGINAL`，大样本在较小 delta 下 `READABLE`；delta 变大又推回 `MARGINAL`。
7. 看真实案例：同一榜面，human floor 可读，32 名 LLM judge floor 不可读。
8. 打开右侧报告：这是 arx 自己渲染的单文件 HTML；页面没有另写一套判决。

## 真实与合成

| 类别 | 说明 |
|---|---|
| 真实 | 每个判决都由 `audit_leaderboard` 在构建时执行；报告由 `report_html` 生成 |
| 合成 | 八个投票矩阵是构造出来的，不是真实模型排名；真实 ChaosNLI 案例只回放审计结果，不重算语料 |

## 文件

```text
demo/
├── payload.py            # 8 个案例 + 敏感性网格 + 真实案例 + Report HTML
├── build_static.py       # 生成 public/cases.json
├── server.py             # stdlib HTTP，静态页 + /api/run
├── public/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── config.json
│   └── cases.json
├── test_demo.py
└── README.md
```

## 边界

- `READABLE` 只表示差距超过测量误差，不表示指标本身是对的。
- 闸审计的是榜面的可读性，不审计模型输出是否忠实、任务是否有意义。
- 合成案例用来行使四种判决，不能当作真实模型排名。
- 真实数据案例见 `case-studies/chaosnli-noise-floor/`；本页不重算它。
- `cases.json` 不参与 `worktree_dirty` 判断。
