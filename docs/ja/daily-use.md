# Persona Image Labの日常運用

普段使う操作は、起動・アクセス・停止の3つです。初回セットアップが済んでいる前提で説明します。

## 1. 使うときだけ起動する

リポジトリのディレクトリで次を実行します。

```bash
./persona start
```

`start` は先にモデルの状態を確認し、画像モデルとPrompt Enhancerを起動します。host側でvLLMが利用できる場合は、PE-T2I / PE-I2IをFP8で並列ロードします。Labとの通信は `cache/prompt-enhancer/*.sock` のUnix socketだけです。

GB10ではモデルの初回ロードに数分かかります。`start` はImage・PE-T2I・PE-I2Iの状態を数十秒ごとに表示し、必要なサービスがすべて `ready` になるまで待ちます。Prompt Enhancerは初回のみFlashInfer等のJIT準備が入る場合があります。

既定の待機上限は600秒です。必要な場合だけ次のように延長できます。

```bash
PERSONA_START_TIMEOUT=900 ./persona start
```

状態だけ確認したい場合は次を使います。

```bash
./persona status
```

Docker側に加えて、次のような状態が表示されます。

```text
t2i: ready (pid ...)
i2i: ready (pid ...)
```

ログを追う場合は次です。

```bash
./persona logs
```

`Model loaded` とローカルURLが表示されれば利用できます。

## 2. 短い日本語の指示から始める

入力は日本語の自然文で構いません。標準ではPrompt Enhancerが短い指示を画像生成向けの具体的なPromptへ展開します。参照画像なしではPE-T2I、Personaや追加参照がある場合はPE-I2Iを自動で使います。

たとえばPersonaを1つ選んで、次の程度から始められます。

```text
ゲームに熱中するキャラクターA
```

Personaを選んだ場合、本人性や参照画像の役割は内部で補います。`visual-canon.md` のような長い設定文を生成Promptへ直接連結しません。必要なら詳細設定から「短い指示を自動で具体化」をOFFにし、入力文をそのままQwen-Image-2.1へ渡せます。

GB10上のFP8 vLLM実測では、PE-T2Iは約30秒、Persona参照画像2枚を使ったPE-I2Iは約43秒でした。出力するPromptの長さで前後します。画像生成そのものに入る前にこの時間が追加され、短い指示から構図・照明・アスペクト比まで補います。

画像サイズは標準で「自動（Promptから判断）」です。Prompt Enhancerが返すアスペクト比を使い、約1MPを基準に32px単位へ丸めます。自分で固定したい場合は正方形・縦長・横長のプリセットかカスタムを選びます。

生成中は結果欄の下に「Prompt具体化中」「生成中 12/40」「残り約○秒」などの状態を常時表示し、完了後も生成時間・サイズ・Seedを残します。進捗表示は生成結果の画像欄へ重ねず、画像欄は完成時だけ更新します。

Prompt Enhancerを使うと、具体化が完了した時点で結果側に「具体化されたプロンプト」が表示されます。左側の入力プロンプトは書き換えないため、元の指示と実際に画像生成へ渡した内容を見比べられます。履歴から生成結果を開いた場合も、その生成で使った具体化後のPromptを確認できます。

プロンプトや設定を間違えた場合は、生成中だけ表示される「停止」を押してください。Prompt Enhancer処理中はvLLMのストリーム接続を閉じ、画像生成へ進みません。denoising中は次のstepで中断します。停止した生成結果やメタデータは履歴へ保存されません。

途中画像のプレビューは提供していません。Qwen-Image-2.1の生のdenoising途中画像は完成直前まで粒状ノイズが目立ち、追加のVAE decodeや結果欄の再描画に対して実用上の利点が小さかったためです。進捗はstep数と残り時間で確認します。

別の画像を生成している間は「最近の生成」をクリックしても表示を切り替えません。生成完了後に結果を確定してから、通常どおり履歴を選べます。

履歴を複数消す場合は「最近の生成」の上にある「選択」を押します。各サムネイルに `☐` が表示されるので、削除したい画像をクリックして `☑` に切り替えます。「選択した画像を削除」から件数を確認してまとめて削除できます。削除は元に戻せません。

Prompt EnhancerはvLLMのStructured OutputでJSON形式を固定しています。まれに出力形式が崩れた場合は、出力上限を広げて低温度で1回だけ自動再試行します。それでも失敗した場合は画像生成を中止し、短い元Promptのまま意図と違う画像を作り始めないようにしています。必要な場合だけ、詳細設定から自動補完をOFFにして直接生成できます。

## 3. PCからTailscale経由で開く

Sparkと普段使うPCの両方が同じTailscale tailnetへ参加していれば、ブラウザから次の形式のURLを開けます。

```text
https://<Sparkのホスト名>.<tailnet名>.ts.net/
```

実際のURLはSpark側で確認できます。

```bash
tailscale serve status
```

Tailscale Serveは `127.0.0.1:7862` のPersona Image Labへ中継します。Docker側の公開範囲をLAN全体へ広げる必要はありません。

## 4. 使い終わったら停止する

```bash
./persona stop
```

Persona Image Labは画像生成時だけ起動する運用を前提にしています。停止するとDockerの画像モデルだけでなく、host側のPE-T2I / PE-I2Iプロセスも終了するため、ローカルLLMなど別の用途へメモリを戻せます。

Tailscale Serveは停止しなくて構いません。Persona Image Labが止まっている間は同じURLへアクセスしてもバックエンドへ接続できませんが、次回 `./persona start` した後はURLを変えずに再利用できます。

## よく使うコマンド

| やりたいこと | コマンド |
| --- | --- |
| 起動する | `./persona start` |
| 停止する | `./persona stop` |
| 状態を見る | `./persona status` |
| ログを見る | `./persona logs` |
| 環境を診断する | `./persona doctor` |
| Prompt Enhancerを取得する | `./persona download-enhancers --accept-model-license` |

現状は `./persona` をリポジトリ内から実行します。どこからでも短いコマンドで起動できる仕組みは、必要性が固まってから追加する方針です。
