# 論文探索システム

[![IMAGE ALT TEXT HERE](https://jphacks.com/wp-content/uploads/2021/07/JPHACKS2021_ogp.jpg)](https://www.youtube.com/watch?v=LUPQFB4QyVo)

## 製品概要
### 論文探し×Tech

> **視覚的で対話的な論文探索システム**

> 論文探索システムは，arXiv のデータベースと AI 技術を活用した論文探しをサポートする Web アプリケーションです．２つのマップと3種類の可視化方法で新しい論文探索体験を提供します．

### 背景(製品開発のきっかけ、課題等）
　論文探し/論文サーベイは研究者や大学院生にとって自分の研究の手がかりを探したり，立ち位置やアピールポイントを確立するために必要不可欠である．

　そのような論文探しは４つの点において難しい．まず，そもそも情報を検索するのに，自分の研究と関連するキーワードや研究分野を知っておかなければいけない．次に，英語で出てきた論文の要点を短時間で把握しなければいけない．そして，一つの論文だけでなく複数の論文を網羅的に把握し，研究の世界観を過不足なく構築する必要がある．最後に，日々投稿されている最新の論文の情報を随時取り入れなければいけない．このように論文探しには難点が多く，これらの難点は若手の研究者や駆け出しの研究者ほど障壁が高く，論文サーベイは膨大な時間がかかる．
 
　論文サーベイは最終的に自分の論文と一番関わる論文（キー論文）を見つけることが目標になるが，手あたり次第論文を読むだけでは途方もない時間がかかり見つけることはできない．読んだ論文を様々なレベルで抽象化し，抽出した研究分野・キーアイディア・**キーワード** を吟味して，自分の研究・研究分野との関係性を把握し，知識を蓄え，仮説を立てながら論文を ***探索*** する必要がある．

### 製品説明（具体的な製品の説明）

論文探索システムは，検索してヒットした論文データを生成モデルでモデリングし，論文探索に必要となるあらゆる情報をユーザーに提供します．
ユーザーは論文マップと単語マップの**２つのマップ**と，そのマップに彩色する**３種類の可視化法**を用途に合わせて選択することで論文に関する様々な解像度の知識を取得します．

### 特長

#### 1. 特長1　検索したキーワードにまつわる論文と単語の類似度の可視化

論文どうし単語どうしの類似度を知ることができる．（さらにU-matrix でそれを補完することができるのじゃ）
ここに，U-matrixの図どーん
#### 2. 特長2　その上で、要素単位で着目した情報の可視化
着目した論文でよく出てくる単語，反対に単語が出てくる論文を知ることができる．（さらにそれが下で表示されて読めるのじゃ）
CCP の図どーん
#### 3. 特長3　その上で、トピック情報の可視化
論文と単語のつながりのクラスタがわかるぞい。
Topicのズドーン

### 解決出来ること
単語マップによりキーワード，クラスタによりジャンルを把握することができる．
着目した論文で出でくる単語を視覚的に把握することで，論文概要の大枠を把握できる．
ひと目で類似する論文，類似しない論文を把握できることで自分の研究の立ち位置の確立を手助けする．
年度を絞って検索できるため，最新の情報も簡単に取り入れられる．

### 今後の展望
論文の被引用関係情報を取り入れてよりリッチな情報を提供する．
単語マップ上で複数の単語を選択できるようにし，AND・OR検索を可能にする．
検索履歴のログを取れるようにして，検索の補完を効かせる．
ユーザーからアンケート収集し，有用性の評価を行う．

### 注力したこと（こだわり等）
* 表示するだけでなく，ユーザーが情報を探索するのを補助する機能
 * 単語マップを見ていて気になった単語を検索キーワードを追加できます．
 * 気になった論文のアブストラクトは本サービスの中で閲覧することができます．
* 情報が煩雑になるところをマップの色合いや見やすさを左右するマップの色合い
* マップの精度を左右するデータの前処理

## 開発技術
### 活用した技術
#### API・データ
* arXiv API（https://arxiv.org/help/api/）


#### フレームワーク・ライブラリ・モジュール
*
* バックエンド
	（可視化）Dash, Plotly
	（前処理）nltk,  sklearn
	（モデル）numpy, jax
* アプリケーション
	Dash
*  デザイン
	Dash
	Bootstrap

#### デバイス
* PC
* 

### 独自技術
[[ #### ハッカソンで開発した独自機能・技術]]
* 独自で開発したものの内容をこちらに記載してください
* 特に力を入れた部分をファイルリンク、またはcommit_idを記載してください。
	
	https://github.com/furukawa-laboratory/ExploreSearchSystem/blob/main/jax_tsom.py
	https://github.com/furukawa-laboratory/ExploreSearchSystem/blob/main/webapp/event_handler.py
	

#### 製品に取り入れた研究内容（データ・ソフトウェアなど）（※アカデミック部門の場合のみ提出必須）
* [Tensor SOM](https://www.sciencedirect.com/science/article/pii/S0893608016000149)
* [トピック分解](野口君)
