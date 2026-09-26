# Fork元の変更は自動同期しない

Persona Image LabはFork関係を維持しています。ただし、Fork元とはUI、Persona管理、Dockerのデータ境界がすでに異なります。

そのため、GitHubのSync forkで `main` をそのまま追従させる運用はしません。

## 変更が来たら差分だけ確認する

```bash
git fetch upstream --prune
git log --oneline main..upstream/main
```

新しいコミットがなければ何もしません。

変更がある場合は、Qwen-Image-2.1の推論、DGX Spark / GB10対応、依存関係、セキュリティ、生成履歴の修正を優先して確認します。

## 必要な修正だけPersona Image Labへ移す

そのまま適用できる小さな修正ならcherry-pickします。構造が変わっている場合は、同じ修正内容をPersona Image Lab側へ実装します。

広範囲なmergeで競合をまとめて解決する方法は避けます。

推論やランタイムに触れた場合は、CPU CIだけで終わらせずGB10実機でも生成確認します。

英語の詳細方針は [../upstream.md](../upstream.md) にあります。
