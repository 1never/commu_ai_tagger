# 日本語対話行為タガー
発話に対して対話行為を付与するライブラリです．

- 旅行代理店タスク対話コーパスに付与された対話行為タグを用いて学習を行っています．
- ISO 24617-2で定義された対話行為タグのサブセット，および旅行代理店タスクのために設計された対話行為タグを付与することができます．

## インストール
必要なライブラリ
```
torch
transformers
fugashi
sentencepiece
unidic-lite
```

インストール方法
```
git clone https://github.com/1never/commu_ai_tagger.git
cd commu_ai_tagger
pip install .
```

## サンプルコード
ISOの対話行為タグを付与する場合
```python
from commu_ai_tagger import Tagger

# modeでisoを指定．
tagger = Tagger(mode="iso", device="cuda:0")

# 対話履歴を入力すると，文脈を考慮した最後の発話に対する対話行為タグを予測する．
context = ["どこか行きたいところはありますか？", "公園に行きたいです．"]

print(tagger.predict(context)) # Answer
```

旅行代理店タスク専用の対話行為タグを付与する場合
```python
from commu_ai_tagger import Tagger

# modeでspecificを指定．
tagger = Tagger(mode="specific", device="cuda:0")

# 対話履歴を入力すると，文脈を考慮した最後の発話に対する対話行為タグを予測する．
context = ["どこか行きたいところはありますか？", "公園に行きたいです．"]

print(tagger.predict(context)) 
```

## タグの一覧


