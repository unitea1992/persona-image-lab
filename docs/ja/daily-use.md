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

## 2. PCからTailscale経由で開く

Sparkと普段使うPCの両方が同じTailscale tailnetへ参加していれば、ブラウザから次の形式のURLを開けます。

```text
https://<Sparkのホスト名>.<tailnet名>.ts.net/
```

実際のURLはSpark側で確認できます。

```bash
tailscale serve status
```

Tailscale Serveは `127.0.0.1:7862` のPersona Image Labへ中継します。Docker側の公開範囲をLAN全体へ広げる必要はありません。

## 3. 使い終わったら停止する

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
