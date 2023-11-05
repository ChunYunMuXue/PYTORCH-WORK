### 分布差异:
#### KL散度 相对熵：
$D_{KL}(P||Q) = \sum_{x\in X}P(x)\log \frac{P(x)}{Q(x)}$
非对称

#### 交叉熵

$\sum P_{D_1}(x)\ln \frac{1}{P_{D_2}(x)}$ 交叉熵越小越相似

#### BCE 损失

$loss = -(y\log \hat y + (1 - y) \log {(1 - \hat  y)})$


