# Personaを追加・更新する

Personaには、キャラクターを識別するためのプロンプトと参照画像をまとめます。

実データは公開リポジトリへ入れません。リポジトリ外の専用ディレクトリへ置くのが基本です。

## Personaの保存先を指定する

`.env.example` を `.env` へコピーし、`PERSONA_DATA_DIR` を設定します。

```dotenv
PERSONA_DATA_DIR=/absolute/path/to/private/personas
```

指定したディレクトリはコンテナ内へ読み取り専用でマウントされます。

`model/`、`outputs/`、`cache/` と同じ場所や、その親子関係になるパスは指定できません。

## 1キャラクターにつき1ディレクトリ作る

```text
private-personas/
└── sample-character/
    ├── persona.json
    ├── visual-canon.md
    ├── identity.png
    └── fullbody.png
```

`persona.json` の最小例です。

```json
{
  "schema_version": 1,
  "name": "Sample Character",
  "description": "Private local character preset",
  "prompt_prefix": "Reference image 1 defines this character's identity. Keep the same person and follow the user's scene request.",
  "prompt_file": "visual-canon.md",
  "references": [
    {"label": "Identity", "path": "identity.png"},
    {"label": "Full body", "path": "fullbody.png"}
  ]
}
```

`prompt_prefix` はQwen-Image-2.1へ実際に渡す短い固定指示です。本人参照が何枚目か、どの画像を本人性や全身比率の基準にするか、といった参照画像の役割を簡潔に書きます。

`prompt_file` には詳しい視覚正本を置けますが、その本文は生成Promptへ連結しません。長いMarkdown設定を画像モデルへそのまま渡すと、文字入りの設定資料として描画されることがあるためです。

## 参照画像は合計10枚まで

Persona側の参照画像と、画面から追加する一時的な参照画像を合わせて最大10枚です。

Persona内のファイルパスは、そのPersonaディレクトリの外を参照できません。

シンボリックリンクを使ったPersonaディレクトリや参照ファイルも拒否されます。

## 更新後は再起動する

Personaの追加や設定変更を画面へ反映するときは、Persona Image Labを再起動します。

```bash
./persona stop
./persona start
```

モデル本体を再ダウンロードする必要はありません。
