# 真面目なプログラマのためのディープラーニング入門

サイト本体: https://euske.github.io/introdl/index.html

このリポジトリは、
「真面目なプログラマのためのディープラーニング入門」
のソースコードです。

## 免責事項

本講座は新山 祐介が個人の興味において制作したものです。
内容の正確さは保証しません。
また本サイトは常時更新されており、内容は予告なく変更されることがあります。

## ライセンス

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="クリエイティブ・コモンズ・ライセンス" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />本講座は、<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">クリエイティブ・コモンズ 表示 - 継承 4.0 国際 ライセンス</a>の下に提供されています。

## 今後の予定

 - 演習の回答を提示・解説するYouTube動画を作成。
 - データ形式について説明する入門的な章を挿入。

## サイト製作の動機

このサイトはもともと、大学の講義で使うための素材として作り始めた。
が、学生にはやや高度な内容も含まれていたため「プログラマ向け」としてある。
基本的には「筆者 (新山) がニューラルネットワークを学び始めたとき、
こういう入門書があったらよかった」と思えるような内容を目指している。

筆者がこれまで大学等で受けてきた機械学習の授業の多くは
「機械学習 (ニューラルネットワーク) とはどういうものか」を
総論として理解することを目的としており「このようなシステムを作りたい」
という目的指向があるものではなかった。また、たいていの
Python を使った機械学習の書籍・講義等では NumPy や PyTorch を
いきなり最初から使っており、筆者の好みではもっと原始的な部分から
解説してほしかったというのもある。

これをふまえて、本講座は「画像認識システムを作る」という目的に特化し、
実践的なシステム設計に使ってもらえるよう、アノテーションツールや
転移学習、ONNX などの周辺技術もカバーすることにした。
