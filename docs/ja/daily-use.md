# Persona Image Labの日常運用

普段使う操作は、起動・アクセス・停止の3つです。初回セットアップが済んでいる前提で説明します。

## 1. 使うときだけ起動する

リポジトリのディレクトリで次を実行します。

```bash
./persona start
```

`start` は先にモデルの状態を確認してからコンテナを起動します。モデルの読み込み中はブラウザへまだ接続できません。GB10実機では、Qwen-Image-2.1の読み込みにおおむね3〜4分かかります。

状態だけ確認したい場合は次を使います。

```bash
./persona status
```

ログを追う場合は次です。

```bash
./persona logs
```

`Model loaded` とローカルURLが表示されれば利用できます。

## 2. プロンプトは日本語の自然文で入力する

Qwen-Image-2.1へは、日本語で場面や服装をそのまま指定できます。Stable Diffusion系でよく使われる単語の羅列へ変換する必要はありません。

たとえばキャラクターを選び、次のように入力します。

```text
公園で立っている。白いワンピース、やわらかい夕方の光、全身。
```

Personaを選んだ場合、本人性や参照画像の役割は内部で補います。`visual-canon.md` のような長い設定文を生成Promptへ直接連結しないため、ユーザーは今回の画像で変えたい内容だけを書けば足ります。

生成中はQwen-Image-2.1のstep進捗が表示されます。途中latentを画像へ戻して逐次表示するには追加のVAEデコードが必要になるため、生成速度とGPUメモリへの影響を避けて現状は使いません。

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

Persona Image Labは画像生成時だけ起動する運用を前提にしています。停止すればモデルを保持していたプロセスも終了するため、ローカルLLMなど別の用途へメモリを戻せます。

Tailscale Serveは停止しなくて構いません。Persona Image Labが止まっている間は同じURLへアクセスしてもバックエンドへ接続できませんが、次回 `./persona start` した後はURLを変えずに再利用できます。

## よく使うコマンド

| やりたいこと | コマンド |
| --- | --- |
| 起動する | `./persona start` |
| 停止する | `./persona stop` |
| 状態を見る | `./persona status` |
| ログを見る | `./persona logs` |
| 環境を診断する | `./persona doctor` |

現状は `./persona` をリポジトリ内から実行します。どこからでも短いコマンドで起動できる仕組みは、必要性が固まってから追加する方針です。
