# Persona Image Lab

[English](README.md)

DGX Spark / GB10 上で動かす、ローカル向けの画像生成UIです。Qwen-Image-2.1を使い、キャラクターごとの設定や参照画像は公開リポジトリの外に置けます。

## 普段は「起動 → ブラウザ → 停止」だけ

現状はリポジトリのディレクトリで操作します。

```bash
./persona start
```

起動後は、同じTailscale tailnet内のPCから次のURLを開きます。

```text
https://<Sparkのホスト名>.<tailnet名>.ts.net/
```

使い終わったら停止します。

```bash
./persona stop
```

Persona Image Labは常駐を前提にしていません。モデルを使わない時間はコンテナを止め、ローカルLLMなど別の用途へメモリを戻します。

Tailscale Serveの設定は残して構いません。次回も同じURLを使えます。

詳しい普段使いは [日常運用ガイド](docs/ja/daily-use.md) を参照してください。

プロンプトは日本語の自然文で入力できます。キャラクターの固定情報はPersona側で扱うため、毎回長い設定を書く必要はありません。場面、服装、表情、構図など、その画像で変えたい内容を普通の文章で指定します。

## 初回だけ必要な準備

```bash
./persona build
./persona doctor
./persona download --accept-model-license
```

別のPCから使う場合は、Spark側でTailscale Serveを一度設定します。

```bash
tailscale serve --bg 7862
tailscale serve status
```

Persona Image Lab自体は `127.0.0.1:7862` にだけ公開します。tailnet内のHTTPSアクセスはTailscale Serveが受け持ちます。

インターネット公開用のTailscale Funnelは使いません。

## Personaデータは公開リポジトリと分離する

実キャラクターの設定や参照画像は、リポジトリ外のディレクトリへ置く運用を推奨します。

```dotenv
PERSONA_DATA_DIR=/absolute/path/to/private/personas
```

このディレクトリはコンテナ内へ読み取り専用でマウントされます。詳しい構成は [Personaの追加方法](docs/ja/personas.md) にまとめています。

## 困ったときに見るページ

- 普段の起動・停止・Tailscaleアクセス: [docs/ja/daily-use.md](docs/ja/daily-use.md)
- Personaの追加・更新: [docs/ja/personas.md](docs/ja/personas.md)
- 接続できない、起動しない: [docs/ja/troubleshooting.md](docs/ja/troubleshooting.md)
- Fork元の変更を取り込む: [docs/ja/upstream.md](docs/ja/upstream.md)

英語の技術資料は `docs/` に残しています。日常利用では、まずこのREADMEと `docs/ja/` だけ見れば足ります。
