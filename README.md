# Reversi_with_Simple_AlphaZero

## Introduction
參考AlphaZero的方法訓練AI玩Reversi

## Reference
- [A Simple Alpha(Go) Zero Tutorial](https://web.stanford.edu/~surag/posts/alphazero.html)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [Alpha Zero General (any game, any framework!)](https://github.com/suragnair/alpha-zero-general)
- [強化學習開發黑白棋、五子棋遊戲](https://blog.csdn.net/a1066196847/article/details/104175799)
- [AlphaZero背后的算法原理解析](https://zhuanlan.zhihu.com/p/78991762)

## Method
- We implemented a simple version alphaZero

### MCTS
- Monte Carlo tree search 是一種在給定 Iteration 次數的情況下將各種 State 以 Tree 的方式建立關聯並進行模擬、搜尋的方法，給定的Iteration 次數越高，決定一步棋的時間就會越久，但對於 State 的評估也會越精準
![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/MCTS_%28English%29.svg/1920px-MCTS_%28English%29.svg.png)

### AlphaZero
- AlphaZero 的前身 AlphaGO 於2016年轟動全世界，2018年可泛用於各種複雜棋類遊戲的AlphaZero問世，其核心概念其實就是利用神經網路來協助MCTS專注於搜索神經網路認為有價值的子樹而忽略沒有價值的子樹，遊戲結束後MCTS再將模擬結果反饋給神經網路相輔相成。

### Our Simple AlphaZero
- AlphaZero使用了相當龐大的神經網路與龐大的運算資源，他們的神經網路最終能針對所有動作輸出機率分布卻能正確識別出合法的動作，我們認為判斷短期無法重現出如此驚人的結果，所以捨棄了動作機率分布的部分，只重現了針對勝率評估，value_base 的部分。
![](https://i.imgur.com/n6dhGit.png)

![](https://i.imgur.com/nfhoWPT.png) ![](https://i.imgur.com/NuVMljM.png) ![](https://i.imgur.com/eaTQAaS.png) ![](https://i.imgur.com/6kIN8ok.png)

## Training

### Structure of NNet
![](https://i.imgur.com/IcjVfXO.png)
### Features
- 2D-array用來表示己方棋子在盤面上的位置，有的位置標一，沒有的標零
- 2D-array用來表示對手棋子在盤面上的位置，有的位置標一，沒有的標零
- 己方棋子數量
- 對手棋子數量
- 場上棋子數量
### Data Flow
代表雙方盤面的Array會先由三層捲積層進行特徵提取，隨後在進入 fully connected layer 時才將棋子數量的Data輸入

### Other Info
- Library: pytorch
- Loss Function: smooth_l1_loss (similar to MSE)
- Optimizer: Adam
- Learning Rate: 0.001 ~ 0.00001
- 訓練的同時不斷讓最好的模型與最新的模型進行對局，若最新的模型勝率超過threshold則將該模型紀錄為最好的模型

## Experiment
:::info
Rule: 
1. 100 games
2. half Black, half White

:::
AI (Iter=1) vs 100% random
win_rate: 55%
![](https://i.imgur.com/3fiFUBy.png)

AI (Iter=10) vs 100% random
win_rate: 71%
![](https://i.imgur.com/DotBGyB.png)

AI (Iter=20) vs 100% random
win_rate: 65%
![](https://i.imgur.com/A3J1HEi.png)

AI (Iter=50) vs 100% random
win_rate: 75%
![](https://i.imgur.com/P0ehYny.png)

AI (Iter=100) vs 100% random
win_rate: 78%
![](https://i.imgur.com/Muei0En.png)

AI (Iter=50) vs AI (Iter=10)
win_rate: 81%
![](https://i.imgur.com/tFZRjpm.png)



## Result
- 在 Iter=1 的實驗中演算法只根據當前盤面進行評估而不會去考慮未來的盤面，即便如此相較完全隨機，我們的 AI 依舊稍占優勢
- 在 Iter\=10 和 Iter\=20 的實驗中勝率很明顯地高於 Iter==1，表示我們的AI在經過探索之後確實可以做出比最快速的方式更好的決策
- 在 Iter\=50 和 Iter\=100 的實驗中，勝率已經遠高於完全隨機的策略，卻又不是與傳統的窮舉所有可能的方法，而是透過神經網路的輔助專注於有價值的盤面的搜索
- 在 Iter\=50 vs Iter\=10 的實驗中，Iter\=50 的AI勝率遠高於 Iter\=10 ，要打比方的話就是 Iter\=50 比對手要更有遠見。
## Conclusion
- 因為沒有一個明確的訓練指標，且大量進行模擬相當耗時，所以訓練起來相當漫無目的且困難。
- 透過此次 Project 理解了 AlphaZero 的厲害之處，核心概念其實僅是將二十年前的演算的核心部分用神經網路代替，就造就了驚動全世界的創舉。
- 該演算法與以往嘗試過的 Reinforcement Learning 極為不同，做出決策的是結合了非AI演算法與AI相輔相成輸出結果的應用。


