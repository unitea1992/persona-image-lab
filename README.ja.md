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

Prompt Enhancerは、host側のvLLMが利用できればPE-T2I / PE-I2IをFP8で起動し、LabとはUnix socketだけで通信します。`./persona stop` ではこれらもまとめて終了します。

Tailscale Serveの設定は残して構いません。次回も同じURLを使えます。

詳しい普段使いは [日常運用ガイド](docs/ja/daily-use.md) を参照してください。

プロンプトは日本語の自然文で入力できます。短い指示は標準でPrompt Enhancerが具体化するため、Stable Diffusion系のタグ列へ書き換える必要はありません。参照画像がない生成はPE-T2I、Personaや参照画像を使う生成はPE-I2Iへ自動で振り分けます。

たとえば `ゲームに熱中するキャラクターA` のような短い指示から始められます。本人性はPersona側で扱うため、毎回キャラクター設定を書き直す必要もありません。

## 初回だけ必要な準備

```bash
./persona build
./persona doctor
./persona download --accept-model-license
./persona download-enhancers --accept-model-license
```

Qwen-Image-2.1本体に加え、PE-T2I / PE-I2Iの2モデルをローカルへ保存します。3モデルともQwen Research Licenseの対象で、非商用の研究・評価用途を前提とします。

検証したGB10環境ではhostのvLLM 0.27.1を利用しています。vLLMが使えない環境ではTransformersへフォールバックできますが、Prompt展開はかなり遅くなります。

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
