# GAMMA 在 RoboMemArena 上的评测（seed 50–99）

**已终止**。已完成 4 / 1300 个回合，从本次评测启动起经过 0.74 小时。

覆盖 26 个任务，每任务 50 个布局，seed 50–99。每次预测执行 10 步，策略上限 2500 步。
GAMMA 使用作者的固定子任务序列和机械核验。原论文的 RoboMemArena 专用模型包仍缺，本次使用 Qwen3.8-27B writer、SAM3 与 RoboMemArena 发布版 π₀.₅。 [论文评测协议](https://arxiv.org/html/2610.12793v1#A7) · [权重说明](https://github.com/1999Lyd/GAMMA/blob/main/CHECKPOINTS.md)

评分使用 GAMMA 论文发表时可用的 RoboMemArena 提交 `d09f63b`。该版倒液评分测试有 4 处断言与实现不一致，实测沿用该提交的评分实现。 [RoboMemArena 提交](https://github.com/OpenHelix-Team/RoboMemArena/commit/d09f63b671f362985e1e0d038d2014c2d2519488)

切换到 `d09f63b` 前有 2 个回合已完成，实际运行的提交为 `cc156e5`。版本差异只涉及任务 5 的评分，保留的回合均来自其他任务。

TSR 是官方任务成功率，CSR 是官方阶段完成率。仿真步数按成功返回的 `env.step` 调用计，含 10 步预热。回合耗时从环境 reset 起算，含推理、排队和仿真。累计耗时是各有效回合之和。

| 任务 | 已完成 / 计划 | 成功 | TSR | CSR | 平均仿真步 | 总仿真步 | 平均耗时 (s) | 累计耗时 (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 · `cookies_tomato_basket` | 1 / 50 | 1 | 100.00% | 100.00% | 1077.0 | 1077 | 783.8 | 783.8 |
| 02 · `butter_popcorn_basket` | 1 / 50 | 1 | 100.00% | 100.00% | 874.0 | 874 | 639.4 | 639.4 |
| 03 · `cream_pudding_basket` | 1 / 50 | 1 | 100.00% | 100.00% | 885.0 | 885 | 595.2 | 595.2 |
| 04 · `drawer_butter` | 0 / 50 | — | — | — | — | — | — | — |
| 05 · `butter_middle_drawer` | 1 / 50 | 0 | 0.00% | 0.00% | 2510.0 | 2510 | 829.3 | 829.3 |
| 06 · `pour_tomato_sauce_into_bowl_drainer` | 0 / 50 | — | — | — | — | — | — | — |
| 07 · `pour_tomato_sauce_into_frypan` | 0 / 50 | — | — | — | — | — | — | — |
| 08 · `pick_chocolate_in_frypan_pour_tomato_sauce_twice` | 0 / 50 | — | — | — | — | — | — | — |
| 09 · `pick_butter_in_frypan_pour_tomato_sauce_twice_and_place_tomato_sauce_in_bowl_drainer` | 0 / 50 | — | — | — | — | — | — | — |
| 10 · `pour_wine_bottle_into_mug` | 0 / 50 | — | — | — | — | — | — | — |
| 11 · `cookies_top_butter_middle` | 0 / 50 | — | — | — | — | — | — | — |
| 12 · `cookies_chocolate_middle_drawer` | 0 / 50 | — | — | — | — | — | — | — |
| 13 · `butter_cookies_middle_drawer` | 0 / 50 | — | — | — | — | — | — | — |
| 14 · `cookies_chocolate_drawers` | 0 / 50 | — | — | — | — | — | — | — |
| 15 · `butter_milk_frypan` | 0 / 50 | — | — | — | — | — | — | — |
| 16 · `pour_milk_in_red_coffee_mug_twice_and_place_milk_in_bowl_drainer` | 0 / 50 | — | — | — | — | — | — | — |
| 17 · `butter_chocolate_middle_drawer` | 0 / 50 | — | — | — | — | — | — | — |
| 18 · `chocolate_butter_cabinet` | 0 / 50 | — | — | — | — | — | — | — |
| 19 · `tomato_milk_orange_cabinet` | 0 / 50 | — | — | — | — | — | — | — |
| 20 · `cookies_chocolate_microwave` | 0 / 50 | — | — | — | — | — | — | — |
| 21 · `butter_chocolate_microwave` | 0 / 50 | — | — | — | — | — | — | — |
| 22 · `pour_tomato_cookies_microwave` | 0 / 50 | — | — | — | — | — | — | — |
| 23 · `cream_popcorn_microwave` | 0 / 50 | — | — | — | — | — | — | — |
| 24 · `cookies_popcorn_microwave` | 0 / 50 | — | — | — | — | — | — | — |
| 25 · `butter_cream` | 0 / 50 | — | — | — | — | — | — | — |
| 26 · `chocolate_pudding_cream` | 0 / 50 | — | — | — | — | — | — | — |

[逐回合数据](episodes.csv) · [运行配置](run_config.json)
