from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    'unidic-lite',
    'fugashi',
    'sentencepiece',
    'transformers',
    'torch',
]

setup(
    name='commu-ai-tagger',  # パッケージ名（pip listで表示される）
    version="0.0.1",  # バージョン
    description="Simple dialogue act tagger",  # 説明
    author='Michimasa Inaba',  # 作者名
    packages=find_packages(),  # 使うモジュール一覧を指定する
    install_requires=INSTALL_REQUIRES,
    license='MIT'  # ライセンス
)