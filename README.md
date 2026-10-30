# 自然言語でスキルの選び方を書いて SO-101 を動かす開発キット ― OSS の判断 AI で、生成 AI のコストを下げる

Physical AI 応用1講座 最終課題の作業リポジトリです。

MuJoCo 上のロボットアーム SO-101 に、3 cm の赤いキューブを持ち上げさせる作業と、緑のゴールへ運んで置かせる作業を行わせます。役割は 3 つに分けています。

| 役割 | 担当 | すること |
|---|---|---|
| 目 | コーディングエージェント（Claude Code、Codex など） | 各ステップの画像を見て、「つかめているか」など 10 項目を yes / no で答える |
| 頭 | 判断 AI（System One 型） | 観測から次の行動（スキル）を確率付きで選び、安全確認の yes / no 確率も返す |
| 手 | プログラム（IK + MuJoCo） | 選ばれたスキルを毎回同じように実行し、順番違いの動作は拒否する |

これまで「頭」には、有料のホスティング型モデル **Jev**（TypeSafe）を使っていました。今回は、2026 年 10 月に無料公開された判断 AI に置き換えられるかを検証しました。

- **[strands-decider](https://github.com/strands-labs/strands-decider)**（AWS の研究チーム）：2B 版、E4B 版
- **H2O-Lightning-4B**（H2O.ai）：判断 AI のベンチマーク JevBench で、公開モデルの中で 1 位。総合点 72.5 で Jev の 71.5 を上回る（[H2O.ai の発表](https://h2o.ai/company/press-media/2026/h2o-lightning-4b-ranks-first-on-jevbench)）

どれも自分の PC（Apple Silicon）で動きます。

- 動画：[video/h2o_pick_place_2x.mp4](video/h2o_pick_place_2x.mp4)（H2O-Lightning-4B で「運んで置く」を行った記録。**2 倍速**、約 1 分 30 秒）／ YouTube：https://www.youtube.com/watch?v=ZC-NKxQikww
- 判断の過程つきの動画（2 倍速）：画面の下に、各手で目（Claude）が「はい」と答えた項目、頭（H2O-Lightning-4B）が選んだスキルと確信度、安全確認の確率を表示します。
  - [video/h2o_pick_place_decisions_2x.mp4](video/h2o_pick_place_decisions_2x.mp4)：今までの位置（L1）
  - [video/h2o_pick_place_L8_decisions_2x.mp4](video/h2o_pick_place_L8_decisions_2x.mp4)：キューブとゴールの両方を動かした位置（L8）
- レポート：[report/report.pdf](report/report.pdf)（同じ内容の HTML 版は [report/report.html](report/report.html)）

## Claude Code / Codex で実行するときの大まかな流れ

このキットは、**Claude Code や Codex などのコーディングエージェントの上で実行**します。エージェントが「目」になって、各ステップの画像を読み、観測 JSON を書き、次のコマンドを実行します。Claude や GPT の API キーを別に用意する必要はありません。

**環境の構築から検証までを、エージェントに任せられます。** このリポジトリを clone したフォルダで Claude Code や Codex に「SO-101 の操作環境を構築して、検証して」と頼みます。そうすれば、エージェントが次の作業を自分で進め、このリポジトリと同じ結果を再現できます。

- Python 環境と依存関係の準備
- 判断 AI サーバーの起動
- テストの実行
- シミュレーションでの実行
- 動画の書き出し

1. このリポジトリを clone し、そのフォルダを Claude Code（または Codex）で開きます。手順は [AGENTS.md](AGENTS.md) に書いてあり、エージェントが自動で読み込みます（Claude Code は [CLAUDE.md](CLAUDE.md) 経由で読み込みます）。
2. 次のように日本語で頼みます。

   > このリポジトリで SO-101 の操作環境を構築して、テストが通ることを確かめて。判断 AI には H2O-Lightning-4B をローカルで起動して使い、pick_place を最初から最後まで実行して検証して。画像を見て観測を書き、終わったら動画も作って。

   環境がすでにあるなら、実行だけを頼むこともできます。

   > 判断 AI は http://127.0.0.1:8741 の H2O-Lightning-4B を使って、pick_place を最初から最後まで実行して。

3. エージェントは `start` → 画像を見る → `step` を繰り返し、`[SUCCESS]` か `[STOP]` で止まります。

**注意点**
- H2O-Lightning-4B の重み（数 GB）のダウンロードと、SO-101 モデルの取得には、インターネット接続が必要です。
- H2O-Lightning-4B の Mac 用サーバーは Apple Silicon 向けです。それ以外の環境では、strands-decider か Jev（`TYPESAFE_API_KEY` が必要）を使います。

今回の結果（下の「結果」）は、Claude Code で実行したものです。

## 背景

Physical AI の初心者が、ロボットに「つかんで運ぶ」を成功させるまでの壁は高いです。VLA に動きを学習させる方法では、データ集め・GPU での学習・カメラ条件の調整が必要になります。

そこで、動きを意味のある**スキル**に分け、生成 AI には「次のスキルを選ぶ」ことだけをさせました。こうすると、**学習なしで、スキルの選び方を自然言語で書くだけで**シミュレーションで成功体験を得られる開発キットになります。用意した 10 個のスキルで足りる作業なら、利用者が書くのは、[claude_jev_controller.py](strands_so101_sim/claude_jev_controller.py) の `TaskSpec` の 4 つだけです。

- 作業の目標
- 各スキルを選ぶ条件
- 目に答えさせる質問
- 安全確認の質問

毎ステップの判断まで大きな生成 AI に任せると、トークン代と待ち時間がかかります。そこで判断は、選択肢から確率付きで選ぶ小さな判断 AI に任せます。2026 年 9 月に Jev が公開されてから、**判断を Jev に任せる取り組みが記事で多く紹介される**ようになりました（[TypeSafe AI 公式ブログ](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[Simon Willison の解説](https://simonwillison.net/2026/Sep/21/jev/)）。ロボットでも、Jev に次の動作を選ばせる例が相次いでいます（[ロボスタの記事](https://robotstart.info/article/2026/09/25/382447.html)、[Zenn「SO-101ロボットアーム 実践ガイド」第21章](https://zenn.dev/oggata/books/41ce1497ddde6a/viewer/1eafeb)）。ただし Jev は有料・非公開です。10 月には同じ形式の OSS の判断 AI も公開されました（[labmemo の strands-decider 解説](https://labmemo.com/aws-strands-decider-2b-open-source-decision-model-guide-2026-10/)）。Jev の代わりに **OSS の判断 AI** を使えれば、API キーも使用料も要らず、自分の PC で回せます。

## スキル

運んで置く作業は 10 個、持ち上げる作業は 5 個（1〜4 と 10）のスキルを順に実行します。動きはプログラムに固定されていて、AI は関節の角度も座標も出しません。

AI が次のどちらかを返すと、プログラムはその場で止まります。

- 順番に合わないスキル
- 確信度 0.45 未満の判断

| スキル | プログラムがすること | 実行前の安全確認（yes 確率 ≥ 0.60） |
|---|---|---|
| 1. `move_above_cube` キューブの上へ | 画像からキューブの位置を求め、開いた指をその 9 cm 上へ | 推定位置が作業範囲外なら停止 |
| 2. `descend` 下ろす | 指をキューブの高さまでまっすぐ下ろす | 3 mm 以上ずれたら停止 |
| 3. `close_gripper` つかむ | 指を閉じて両側からはさむ | 両指が触れていなければ停止 |
| 4. `lift` 持ち上げる | 6.5 cm 持ち上げて保つ | 「両方の指でしっかりはさめているか」 |
| 5. `secure_grasp` 持ち直す | 緩んだ握りを空中で少し締め直す | 「空中で、指の間に持てているか」 |
| 6. `transfer` 運ぶ | 高さを保ってゴールの上へ水平に動かす | 両指で持っていなければ停止 |
| 7. `lower` 置く | ゴールの床まで下ろす | 「キューブがゴールの上にあるか」 |
| 8. `release` 離す | 指を開く | 「ゴールの中の床に置かれているか」 |
| 9. `retreat` 離れる | 指を 6.5 cm 上へ離す | — |
| 10. `done` 完了 | 物理的な成功条件を確かめて終了 | 「キューブだけがゴールの中にあり、何も触れていないか」 |

## 結果

### 1. 記録した質問をそのまま聞き直した比較

Jev で実行したときに送った 19 回分の質問を、各モデルにそのまま送りました。質問は次の 3 つの実行記録に入っています。

| 実行記録 | 回数 |
|---|---|
| [持ち上げる・成功](outputs/claude_jev_cube_lift_20260927_184643/session.json) | 5 回 |
| [運んで置く・成功](outputs/claude_jev_pick_place_20260927_185507/session.json) | 10 回 |
| [わざと失敗させた確認（つかめていないと伝えた）](outputs/claude_jev_cube_lift_negative_grasp_check/session.json) | 4 回（拒否された 1 回を含む） |

各記録の `decisions`（と `rejected`）の 1 件が 1 回分の質問です。中身は次のとおりです。

- `jev_state`：判断 AI に送った状態
- `claude_observation`：目の観測
- `action` / `probabilities`：Jev の答え

質問文（スキルを選ぶ条件や安全確認の文）は、[claude_jev_controller.py](strands_so101_sim/claude_jev_controller.py) の `TaskSpec` から組み立てています。各モデルの答えは、[decider_h2o_vs_jev_comparison.json](outputs/decider_h2o_vs_jev_comparison.json) などの `recorded` にあります。

| | Jev | strands-decider 2B | strands-decider E4B | H2O-Lightning-4B |
|---|---|---|---|---|
| 合格ラインを越えた手（成功ランの 15 手中） | 15 | 4 | 4 | **14** |
| Jev と同じ行動（19 手中） | — | 15 | 14 | 18 |
| 行動選択の確信度（中央値） | ≈1.0 | 0.41 | 0.23 | 0.94 |
| 失敗を伝えたときに止まれた場所（7 か所中） | 7 | 7 | 7 | 7 |
| 1 回の応答時間（中央値） | 約 0.2 s | 約 0.5 s | 約 1.5 s | 約 0.9 s |

合格ラインは次の 2 つです。

- 行動選択の確信度が 0.45 以上
- 安全確認の yes 確率が 0.60 以上

### 2. 指示文の修正（H2O-Lightning-4B が手順を飛ばした原因）

H2O-Lightning-4B は、持ち上げた直後に「持ち直す（secure_grasp）」を飛ばして、「運ぶ（transfer）」を選びました。

原因は、元の行動説明にありました。「持ち直す」の説明は「持ち上げた直後で空中にある」、「運ぶ」の説明は「しっかり空中に持っていて、ゴールの上にはない」でした。持ち上げた直後の画像はこの両方に当てはまり、「しっかり」かどうかは画像からは分かりません。

そこで、指示文を 2 か所直しました（[claude_jev_controller.py](strands_so101_sim/claude_jev_controller.py)）。

1. 順番の説明に「1 つずつ順番どおりに進む。目的が済んでいるように見えても飛ばさない」と書き足しました。
2. 「持ち直す」と「運ぶ」を、見た目ではなく、**直前に完了したスキル**で区別する文にしました。

持ち上げた直後に各モデルがつけた確率です。

| | 直す前 Jev | 直す前 H2O | 直した後 Jev | 直した後 H2O |
|---|---|---|---|---|
| 「持ち直す」（正しい手） | 0.79〜0.87 | 0.026 | 1.00 | **0.864** |
| 「運ぶ」 | 0.13〜0.21 | 0.949 | 0.00 | 0.111 |
| 記録した質問で合格した手（15 手中） | 15 | 14 | 15 | **15** |

### 3. シミュレーションでの実行（目：Claude、頭：H2O-Lightning-4B）

| 作業 | 結果 |
|---|---|
| 持ち上げる（5 cm 以上で合格） | 成功：5.80 cm |
| 運んで置く・指示文の修正前 | 5 手目で停止（順番違いを安全装置が拒否） |
| 運んで置く・指示文の修正後 | **成功**：ゴール中心から 7.87 mm（10 mm 以内で合格）。10 手すべて H2O-Lightning-4B が判断 |

### 4. より大きな判断 AI（elephant-1 / mammoth-1、OpenRouter 経由）

10 月 21 日に Baseten の研究チームが公開した [elephant-1](https://huggingface.co/baselabs/elephant-1)（9B）と mammoth-1（27B）を、[OpenRouter の Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request) 経由で、同じ方法で確かめました。説明文は 2. で直したものを使っています。

| | Jev | H2O-Lightning-4B | elephant-1 | mammoth-1 |
|---|---|---|---|---|
| 合格ラインを越えた手（15 手中） | 15 | 15 | 15 | 15 |
| Jev と同じ行動（19 手中） | — | 19 | 19 | 19 |
| 持ち上げた直後の「持ち直す」（元の説明文 → 直した説明文） | 0.79〜0.87 → 1.00 | 0.026 → 0.864 | 0.059 → 0.759 | 0.936 → 0.995 |
| 失敗を伝えたときに止まれた場所（7 か所中） | 7 | 7 | 7 | 7 |
| 運んで置く（シミュレーション） | 成功 7.87 mm | 成功 7.87 mm | 成功 7.87 mm | 成功 7.87 mm |
| 1 回の応答時間（中央値） | 約 0.2 s | 約 0.9 s | 約 0.4 s | 約 1.2 s |

mammoth-1 は元の説明文のままでも正しい順番を選べました。elephant-1 は速い一方、元の説明文では H2O-Lightning-4B と同じく「運ぶ」を選びました（0.93）。H2O-Lightning-4B の失敗 7 か所と応答時間は元の説明文で測った値です。記録は `outputs/decider_{elephant,mammoth}_vs_jev_comparison.json` と `outputs/claude_{elephant,mammoth}/` にあります。

### 5. 位置を変えた 9 通りでの実行（成功率）

キューブとゴールの位置を変えて、運んで置く作業を 9 通り実行しました（目：Claude、頭：H2O-Lightning-4B v1.2.3）。

- 9 通りすべてで成功しました。10 手すべてを、正しい順番で選びました。
- いちばん低い値は、スキル選択の確信度が 0.79、安全確認が 0.80 でした。合格ラインは 0.45 と 0.60 です。
- 置いた位置のずれは 7.75〜8.70 mm でした（合格は 10 mm 以内）。

先に AI なしでスキルを決まった順番に実行する「下見」（`check_layouts.py`）をしました。その結果、下見した 41 通りのうち、作業を終えられたのは 20 通りでした。手（スキルと IK）が作業を終えられる位置は、キューブで x 0.21〜0.22 m、y ±0.02 m の付近だけでした。9 通りは、その中から選んでいます。

シミュレーションは毎回同じ動きになるので、確かめたのは「位置が変わっても目と頭が正しく判断できること」です。各位置は 1 回ずつです。

記録：`outputs/h2o_layout_success_rate.json`、`outputs/claude_h2o_layouts/`、`outputs/layout_feasibility.json`。

次は実機の SO-101 での検証を進めます。[ABEJA の記事](https://tech-blog.abeja.asia/entry/jev-zeroshot-202610)では、大きなモデルによる計画と、Jev 形式の判断 AI による手首カメラでの位置合わせを組み合わせ、実機アームを追加学習なしで動かしています。この記事を参考に、手首カメラでの位置合わせと、見失ったときのやり直しを取り入れる予定です。

## 追記：Jev 1.14 と H2O-Lightning-4B v1.4 の画像入力

Jev 1.14（2026-10-14）で、画像の入力と、物の位置を枠で答える Box 型が使えるようになりました（プレビュー機能）。そこで、記録した 15 枚の画像で、Jev 1.14 に「目」を任せられるかを確かめました。

- 観測が Claude と一致した割合は 81.5〜84.6% でした（3 回実行）。1 枚 0.3〜0.4 秒です。
- 見えるか、浮いているか、ゴールの中かは、3 回とも全問正解でした。
- 指の開き閉じは見分けられませんでした（53〜62%）。そのため、運んで置く作業では「しっかりつかめたか」の安全確認で止まります。

H2O-Lightning-4B も v1.4 で画像と Box に対応しました。同じ 15 枚で確かめた結果は次のとおりです。

- 一致率は 83.8% でした。危ない向きの「はい」は 1 個だけでした（Jev 1.14 は 12〜14 個）。
- キューブの位置は IoU 0.83 と正確で、隠れたキューブも見分けられました。
- 手元の Mac では、画像 1 枚に約 1 分かかりました。
- 新しい重み（v1.4.1）でも、6 章の文章での判断の結果（19/19、15/15、7/7）は変わりませんでした。

詳しくは [report/report.pdf](report/report.pdf) の 10 章を見てください。実行は次のとおりです。

- Jev：`.venv/bin/python eval_eye_models.py jev`（`.env` に `TYPESAFE_API_KEY` が必要です）
- H2O：`.venv/bin/python eval_eye_models.py h2o`（v1.4.2 のサーバーを 8741 番で起動しておきます）

## ファイル構成

```
AGENTS.md / CLAUDE.md         エージェント（Codex / Claude Code）向けの実行手順
run_claude_jev_cube_lift.py   1 判断ずつ進める実行スクリプト（start / step / video）
compare_decider_vs_jev.py     記録した質問の再送と、失敗観測での安全確認（Jev と判断 AI の比較）
compare_prompt_order.py       指示文 3 種（original / strict / strict_anchored）の比較
check_layouts.py              AI なしでスキルを順番に実行し、手が作業を終えられる位置を調べる
eval_eye_models.py            画像を読む「目」の評価（Jev 1.14 / H2O-Lightning-4B v1.4 の画像入力と Box、Ollama の小型モデル）
strands_so101_sim/
  claude_jev_controller.py    質問の組み立て、ゲート（0.45 / 0.60）、Jev / ローカル判断 AI の切り替え
  decider_client.py           /v1/systemone を呼ぶローカル判断 AI 用クライアント（標準ライブラリのみ）
  cube_lift_env.py            持ち上げ作業の環境とスキル、成功判定、遷移チェック
  pick_place_env.py           運んで置く作業の環境とスキル
  so101_pick_ik.py            MuJoCo シーン、IK（mink）、カメラ画像
  cube_vision.py              move_above_cube の目標位置を画像から求める処理
  tests/                      単体テスト（unittest）
outputs/
  claude_h2o/                 H2O-Lightning-4B での実行記録（各ステップの画像、Claude の観測、確率）
  claude_jev_*/               比較の元になった Jev での実行記録
  decider_*_vs_jev_comparison.json, prompt_order_comparison_*.json   比較結果
report/                       レポート（PDF / HTML）
```

各実行記録の `session.json` には、次の内容が入っています。

- Claude の観測
- 判断 AI に送った状態
- 10 個の行動それぞれの確率
- 安全確認の確率
- 応答時間

## 再現方法

確認した環境は macOS（Apple Silicon）、Python 3.13 です。

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
```

SO-101 のモデルは、strands-robots が初回起動時に `robot_descriptions` から取得します。インターネット接続が必要です。

### 判断 AI のサーバー

**H2O-Lightning-4B（Mac 向け）**
コミュニティ製の MLX 版を使いました：[internet-projects/h2o-lightning-mlx](https://github.com/internet-projects/h2o-lightning-mlx)、コミット `efe968e`。重みは 8bit 版の `netprojects/h2o-lightning-4b-mlx-q8` です。

この MLX 版は、2026 年 10 月 12 日に初めて公開されたばかりの、有志による移植版です。その後も更新が続いています。

- **v1.2.3**（最初の版）：6 章までの結果は、この版で測りました。`git checkout v1.2.3` で使えます。
- **v1.4.2**（2026-10-29）：Box に対応し（v1.4.0 から）、重みも新しくなりました（v1.4.1 から）。10 章の結果は、この版で測りました。

```bash
git clone https://github.com/internet-projects/h2o-lightning-mlx h2o-lightning-mlx-v1.2.3
cd h2o-lightning-mlx-v1.2.3
git checkout v1.2.3   # 10 章を試すときは v1.4.2（別のフォルダに clone する）
bash setup.sh q8      # 公式の重み（約 9 GB）を取得し、8bit 版に変換する。setup.sh には実行権限がないので bash で実行する
MODEL=mlx-q8 SHIM_NOUL_FLOOR=0 bash serve.sh   # http://127.0.0.1:8741/v1/systemone
```

`SHIM_NOUL_FLOOR=0` は必須です。v1.2.3 の出荷時の設定（0.801）のままだと、迷っている yes / no の確率を 0.801 / 0.199 に書き換えて返します。そうなると 0.60 の安全確認が意味をなさなくなります。v1.4.2 は初期設定で書き換えませんが、同じように付けておきます。

**strands-decider**
[strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) の GitHub main（`930e6d5`）を別の環境に入れました。

```bash
strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 --device mps --port 8765
```

**OpenRouter の判断 AI（elephant-1 / mammoth-1 など）**
サーバーは要りません。`.env` に `OPENROUTER_API_KEY` を入れ、送り先とモデル名を指定します。無料枠は 1 日 50 回までです。

```bash
export DECIDER_URL=https://openrouter.ai/api/alpha/decisions
export DECIDER_MODEL=baseten/mammoth-1:free     # または baseten/elephant-1:free
```

**Jev**
`.env` に `TYPESAFE_API_KEY` を設定します。有料の API です。

### 1 判断ずつ実行する

目の役は、コーディングエージェント（今回は Claude Code）が担います。画像（`NN_*.png`）を読んで観測 JSON を書き、次の `step` に渡します。観測項目と JSON の書き方は [AGENTS.md](AGENTS.md) にあります。Claude API は使っていません。人が画像を見て同じ JSON を書いても、同じように動かせます。

```bash
export DECIDER_URL=http://127.0.0.1:8741        # 未設定なら Jev を使う
.venv/bin/python run_claude_jev_cube_lift.py start --task pick_place --output-dir outputs/claude_h2o
# → 00_ready.png を見て観測を書く
.venv/bin/python run_claude_jev_cube_lift.py step --session outputs/claude_h2o/<セッション> \
  --observation '{"cube_visible":"yes","goal_visible":"yes","gripper_above_cube":"no", ... ,"notes":"..."}'
# done まで繰り返す。終わったら動画を書き出す
.venv/bin/python run_claude_jev_cube_lift.py video --session outputs/claude_h2o/<セッション> --label "CLAUDE+H2O-LIGHTNING-4B"
# 判断の過程（観測・選んだスキル・確信度・安全確認）を画面の下に出すときは --show-decisions を付ける
.venv/bin/python run_claude_jev_cube_lift.py video --session outputs/claude_h2o/<セッション> --label "H2O-Lightning-4B" --show-decisions
```

各 `step` は、それまでのスキルを最初から再実行します。そのうえで、指先の軌跡が記録と 1e-6 m 以内で一致することを確かめてから、次へ進みます。MuJoCo と IK は決定的です。

### 比較スクリプト

```bash
DECIDER_URL=http://127.0.0.1:8741 .venv/bin/python compare_decider_vs_jev.py decider_h2o_vs_jev_comparison.json
DECIDER_URL=http://127.0.0.1:8741 .venv/bin/python compare_prompt_order.py strict_anchored
.venv/bin/python -m unittest strands_so101_sim/tests/test_claude_jev.py strands_so101_sim/tests/test_decider_client.py strands_so101_sim/tests/test_pick_place.py
```

比較スクリプトは Jev とローカル判断 AI の両方を呼ぶので、`TYPESAFE_API_KEY` が必要です。

## 使用した外部のもの

| 名前 | 用途 | ライセンス |
|---|---|---|
| [strands-labs/robots](https://github.com/strands-labs/robots) | SO-101 モデルの読み込み（pip で取得。このリポジトリには含めていません） | Apache-2.0 |
| [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) | 判断 AI（2B / E4B） | Apache-2.0 |
| [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b)、[internet-projects/h2o-lightning-mlx](https://github.com/internet-projects/h2o-lightning-mlx) | 判断 AI と Mac 用サーバー | Apache-2.0 |
| MuJoCo、mink | 物理シミュレーション、IK | Apache-2.0 |
| [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1)、mammoth-1 | 判断 AI（OpenRouter 経由で利用） | MIT |
| TypeSafe Jev | 比較の基準にした有料判断 AI | 商用 API |
