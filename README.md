# 日本語対話行為タガー
発話に対して対話行為を付与するためのライブラリです．

- [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)に付与された対話行為タグを用いて学習を行っています．
- ISO 24617-2で定義された対話行為タグのサブセット，および旅行代理店タスクのために設計された対話行為タグを付与することができます．
- [LINE DistilBERT](https://huggingface.co/line-corporation/line-distilbert-base-japanese)をファインチューニングしたモデルを使用しています．

## インストール
実行に必要なPythonパッケージ
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
print(tagger.predict(context))  # SpotRequirement
```

## タグの一覧
[ISOタグ](https://github.com/1never/commu_ai_tagger/blob/main/ISO_TAG.md)  
[専用タグ](https://github.com/1never/commu_ai_tagger/blob/main/SPECIFIC_TAG.md)

## 文献情報
本ライブラリを使用した場合は，以下の文献の引用をお願いします．
```
@inproceedings{inaba-etal-2022-collection,
    title = "Collection and Analysis of Travel Agency Task Dialogues with Age-Diverse Speakers",
    author = "Inaba, Michimasa  and
      Chiba, Yuya  and
      Higashinaka, Ryuichiro  and
      Komatani, Kazunori  and
      Miyao, Yusuke  and
      Nagai, Takayuki",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    year = "2022",
    pages = "5759--5767"
}
```

