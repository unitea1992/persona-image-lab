# 困ったときの確認順

上から順に確認してください。最初からログを全部読む必要はありません。

## TailscaleのURLを開けない → まずアプリが起動しているか確認する

```bash
./persona status
```

コンテナが止まっていれば正常です。必要なときだけ `./persona start` してください。

起動直後はモデル読み込みに時間がかかります。GB10実機では3〜4分程度が目安です。

## 起動中なのに画面が出ない → ログでモデル読み込みを確認する

```bash
./persona logs
```

`Model loaded` がまだ出ていなければ、そのまま読み込み完了まで待ちます。

エラーが出ている場合は、先に `./persona doctor` を実行してください。

## Tailscale側だけつながらない → Serveとtailnet接続を確認する

Spark側:

```bash
tailscale serve status
```

PC側:

```bash
tailscale ping <Sparkのホスト名>
```

Serveを初めて設定した直後は、HTTPS証明書の発行に少し時間がかかることがあります。

設定直後だけ接続できない場合は、30〜60秒ほど後にもう一度試します。

## メモリをローカルLLMへ戻したい → Persona Image Labを停止する

```bash
./persona stop
```

Tailscale Serveまで止める必要はありません。Serve自体は画像モデルを読み込まないため、普段は設定を残して構いません。

## 最後にまとめて診断する

```bash
./persona doctor
```

アーキテクチャ、GPU、空き容量、モデル、Persona設定をまとめて確認できます。
