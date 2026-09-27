# Persona Image Labの日常運用

普段使う操作は、起動・アクセス・停止の3つです。初回セットアップが済んでいる前提で説明します。

## 1. 使うときだけ起動する

リポジトリのディレクトリで次を実行します。

```bash
./persona start
```

`start` は先にモデルの状態を確認し、画像モデルとPrompt Enhancerを起動します。host側でvLLMが利用できる場合は、PE-T2I / PE-I2IをFP8で並列ロードします。Labとの通信は `cache/prompt-enhancer/*.sock` のUnix socketだけです。

GB10ではモデルの初回ロードに数分かかります。Prompt Enhancerは初回のみFlashInfer等のJIT準備が入る場合があります。短文補完を確実に使うなら、`./persona status` で両方が `ready` になってから生成します。

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

生成中はstep進捗と途中画像が表示されます。途中プレビューは標準ONで、40 stepならおおむね1/3と2/3の2回更新します。高解像度ではプレビューだけ512px級へ縮小してVAE decodeするため、完成画像の解像度は変えずに負荷を抑えます。不要なら詳細設定からOFFにできます。

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
