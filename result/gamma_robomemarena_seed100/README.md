# GAMMA 在 RoboMemArena 上的评测（seed 100）

**已完成**。已完成 26 / 26 个回合，从本次评测启动起经过 1.91 小时。

覆盖 26 个任务，每任务 1 个布局，seed 100。每次预测执行 10 步，策略上限 2500 步。
GAMMA 使用作者的固定子任务序列和机械核验。原论文的 RoboMemArena 专用模型包仍缺，本次使用 Qwen3.8-27B writer、SAM3 与 RoboMemArena 发布版 π₀.₅。 [论文评测协议](https://arxiv.org/html/2610.12793v1#A7) · [权重说明](https://github.com/1999Lyd/GAMMA/blob/main/CHECKPOINTS.md)

评分使用 GAMMA 论文发表时可用的 RoboMemArena 提交 `d09f63b`。该版倒液评分测试有 4 处断言与实现不一致，实测沿用该提交的评分实现。 [RoboMemArena 提交](https://github.com/OpenHelix-Team/RoboMemArena/commit/d09f63b671f362985e1e0d038d2014c2d2519488)

本次使用演示训练布局。每任务的 TSR 由单个回合决定。

TSR 是官方任务成功率，CSR 是官方阶段完成率。仿真步数按成功返回的 `env.step` 调用计，含 10 步预热。回合耗时从环境 reset 起算，含推理、排队和仿真。累计耗时是各有效回合之和。

| 任务 | 已完成 / 计划 | 成功 | TSR | CSR | 仿真步 | 耗时 (s) |
|:---|---:|---:|---:|---:|---:|---:|
| 01 · `cookies_tomato_basket` | 1 / 1 | 1 | 100.00% | 100.00% | 1122 | 990.6 |
| 02 · `butter_popcorn_basket` | 1 / 1 | 1 | 100.00% | 100.00% | 909 | 639.2 |
| 03 · `cream_pudding_basket` | 1 / 1 | 1 | 100.00% | 100.00% | 880 | 553.5 |
| 04 · `drawer_butter` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 1223.8 |
| 05 · `butter_middle_drawer` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 2213.1 |
| 06 · `pour_tomato_sauce_into_bowl_drainer` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 2176.1 |
| 07 · `pour_tomato_sauce_into_frypan` | 1 / 1 | 1 | 100.00% | 100.00% | 818 | 659.3 |
| 08 · `pick_chocolate_in_frypan_pour_tomato_sauce_twice` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 1648.8 |
| 09 · `pick_butter_in_frypan_pour_tomato_sauce_twice_and_place_tomato_sauce_in_bowl_drainer` | 1 / 1 | 0 | 0.00% | 66.67% | 2510 | 1998.2 |
| 10 · `pour_wine_bottle_into_mug` | 1 / 1 | 1 | 100.00% | 100.00% | 855 | 750.5 |
| 11 · `cookies_top_butter_middle` | 1 / 1 | 0 | 0.00% | 40.00% | 2510 | 2120.7 |
| 12 · `cookies_chocolate_middle_drawer` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 1684.0 |
| 13 · `butter_cookies_middle_drawer` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 1654.1 |
| 14 · `cookies_chocolate_drawers` | 1 / 1 | 0 | 0.00% | 80.00% | 2510 | 1845.4 |
| 15 · `butter_milk_frypan` | 1 / 1 | 1 | 100.00% | 100.00% | 2399 | 2221.0 |
| 16 · `pour_milk_in_red_coffee_mug_twice_and_place_milk_in_bowl_drainer` | 1 / 1 | 0 | 0.00% | 66.67% | 2510 | 1972.2 |
| 17 · `butter_chocolate_middle_drawer` | 1 / 1 | 0 | 0.00% | 66.67% | 2510 | 1884.0 |
| 18 · `chocolate_butter_cabinet` | 1 / 1 | 1 | 100.00% | 100.00% | 1168 | 1029.0 |
| 19 · `tomato_milk_orange_cabinet` | 1 / 1 | 0 | 0.00% | 0.00% | 2510 | 1765.4 |
| 20 · `cookies_chocolate_microwave` | 1 / 1 | 1 | 100.00% | 100.00% | 848 | 500.2 |
| 21 · `butter_chocolate_microwave` | 1 / 1 | 1 | 100.00% | 100.00% | 980 | 710.6 |
| 22 · `pour_tomato_cookies_microwave` | 1 / 1 | 1 | 100.00% | 100.00% | 499 | 374.0 |
| 23 · `cream_popcorn_microwave` | 1 / 1 | 1 | 100.00% | 100.00% | 959 | 520.4 |
| 24 · `cookies_popcorn_microwave` | 1 / 1 | 0 | 0.00% | 33.33% | 2510 | 1634.4 |
| 25 · `butter_cream` | 1 / 1 | 1 | 100.00% | 100.00% | 992 | 622.7 |
| 26 · `chocolate_pudding_cream` | 1 / 1 | 1 | 100.00% | 100.00% | 1005 | 738.8 |

全量 TSR **50.00%**，CSR **63.59%**。

[运行配置](run_config.json)
