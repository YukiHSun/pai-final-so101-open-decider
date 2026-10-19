# エージェント向け手順（Claude Code / Codex など）

このリポジトリでは、コーディングエージェント自身が「目」の役を担います。エージェントは画像ファイルを読み、観測 JSON を書き、次のコマンドを実行します。判断（次のスキル）は判断 AI が、動作はプログラムが行います。関節角や座標を自分で決めてはいけません。

## 準備

- 依存関係は README の「再現方法」に従って `.venv` に入れます。
- 判断 AI を決めます。
  - ローカルの判断 AI を使う場合は `export DECIDER_URL=http://127.0.0.1:8741` を設定します（H2O-Lightning-4B の例。サーバーは README の手順で起動します）。
  - `DECIDER_URL` が未設定なら、`.env` の `TYPESAFE_API_KEY` を使って Jev を呼びます。

## 1 回の実行の流れ

1. セッションを始めます。表示された `[SESSION]` のパスを覚えておきます。

   ```bash
   .venv/bin/python run_claude_jev_cube_lift.py start --task pick_place --output-dir outputs/runs
   ```

2. `[IMAGE]` に表示された PNG を開いて見ます。左が斜めのカメラ、右が真上のカメラです。
3. 下の観測項目すべてに `"yes"` / `"no"` / `"unclear"` で答え、`notes` に見えたことを 1〜2 文（英語、500 文字以内）で書きます。推測で `yes` にせず、見えないときは `unclear` にします。
4. その JSON を渡して 1 手進めます。

   ```bash
   .venv/bin/python run_claude_jev_cube_lift.py step --session <SESSION のパス> --observation '<観測 JSON>'
   ```

5. `[JEV]` の行に、選ばれたスキルと確率が出ます。新しい `[IMAGE]` が出たら 2 に戻ります。
6. `[SUCCESS]` が出たら完了です。`[STOP]` が出たら、そこで止めて理由を報告します。観測を書き換えて先へ進めようとしてはいけません。
7. 終わったら動画を書き出します。

   ```bash
   .venv/bin/python run_claude_jev_cube_lift.py video --session <SESSION のパス> --label "CLAUDE+H2O-LIGHTNING-4B"
   ```

## 観測項目

`--task pick_place`（運んで置く）の 10 項目です。

| キー | 質問 |
|---|---|
| `cube_visible` | 赤いキューブが画像に見えるか |
| `goal_visible` | 緑のゴールが画像に見えるか |
| `gripper_above_cube` | グリッパーがキューブの真上、またはキューブの位置にあるか |
| `fingers_around_cube` | 両方の指が、キューブの高さでキューブを左右からはさむ位置にあるか |
| `cube_between_closed_fingers` | キューブが閉じた指にはさまれているか |
| `cube_lifted_clear` | キューブが床からはっきり浮いているか |
| `cube_above_goal` | キューブが緑のゴールの上にあるか |
| `cube_on_floor_in_goal` | キューブがゴールの中の床に置かれているか |
| `fingers_open` | 指が開いているか |
| `gripper_clear_of_cube` | グリッパーがキューブから離れ、触れていないか |

`--task cube_lift`（持ち上げる）は 6 項目です：`cube_visible`、`cube_on_floor`（キューブが床に置かれているか）、`gripper_above_cube`、`fingers_around_cube`、`cube_between_closed_fingers`、`cube_lifted_clear`。

項目の正式な英語の質問文は、[strands_so101_sim/claude_jev_controller.py](strands_so101_sim/claude_jev_controller.py) の `observation_fields` にあります。

## 観測 JSON の例（持ち上げた直後）

```json
{"cube_visible":"yes","goal_visible":"yes","gripper_above_cube":"yes","fingers_around_cube":"yes",
 "cube_between_closed_fingers":"yes","cube_lifted_clear":"yes","cube_above_goal":"no",
 "cube_on_floor_in_goal":"no","fingers_open":"no","gripper_clear_of_cube":"no",
 "notes":"The red cube has just been lifted and is held between the closed fingers, clearly above the floor. It is still beside the green goal, not over it."}
```

## 作業を変えたいとき

作業の目標・スキルを選ぶ条件・観測の質問・安全確認の質問は、どれも自然言語で `claude_jev_controller.py` の `TaskSpec` に書かれています。判断がおかしいときは、まずこの文を見直します。見た目では区別できない言葉（「しっかり」など）に頼らず、「直前に完了したスキル」のような確かめられる条件で書きます。
