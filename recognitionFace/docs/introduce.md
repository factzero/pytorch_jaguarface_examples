## 1  损失函数

### 1.1  softmax

$loss$公式：
$$
L = -\frac{1}{N} \sum_{i=1}^N log\frac{e^{f_{y_i}}}{\sum_{j}^n e^{f_j}}
$$
其中，$N$是样本数量(batch_size)；$n$是类别数量；$i$是第$i$个样本；$j$是第$j$个类别；$f_{y_i}$是第i个样本对应类别的分数，$f_{y_i} = W_{y_i}^T x_i$；$f_j$是第j个类别的分数，$f_j = W_{j}^T x_i = ||W_j||||x_i|| \cos(\theta_j)$

### 1.2  ArcFace

$loss$公式：
$$
L = -\frac{1}{N} \sum_{i=1}^N log\frac{e^{s.(\cos(\theta_{y_i}+m))}}{e^{s.(\cos(\theta_{y_i}+m))}+\sum_{j=1,j\neq y_i}^ne^{s.\cos\theta_j}}
$$
​        其中，$N$是样本数量(batch_size)，$n$是类别数量

公式形成分两步：

step1：$X$ and $W$ 归一化，即代码中用$L2$正则化处理
$$
\left. \begin{array}\\
{L_1} = -\frac{1}{N} \sum_{i=1}^N log\frac{e^{W_{y_i}^T x_i +b_{y_i}}}{\sum_{j=1}^n e^{W_{j}^T x_i + b_j}} \\
W_{j}^T x_i = ||W_j||.||x_i|| \cos \theta_j = \cos \theta_j \\
b_j = 0
\end{array} \right\} 
\Rightarrow L_2 = -\frac{1}{N} \sum_{i=1}^N log\frac{e^{s.\cos \theta_{y_i}}}{e^{s.\cos \theta_{y_i}}+\sum_{j=1,j\neq y_i}^ne^{s.\cos\theta_j}}
$$
step2：增加角度界限
$$
L_3 = -\frac{1}{N} \sum_{i=1}^N log\frac{e^{s.(\cos(\theta_{y_i}+m))}}{e^{s.(\cos(\theta_{y_i}+m))}+\sum_{j=1,j\neq y_i}^ne^{s.\cos\theta_j}}
$$
注：余弦公式：$\cos(\alpha + \beta)=\cos\alpha.\cos\beta - \sin\alpha.\sin\beta$

所以$loss$公式中$\cos$可展开：$\cos(\theta_{y_i}+m)=\cos\theta_{y_i}.\cos m-\sin\theta_{y_i}.\sin m$

代码解析：

1、$loss$公式必须是单调的，也即$\cos$函数必须是单调的，需要满足限制条件$(\theta + m)\in [0, \pi], 0\leq \theta+m\leq \pi$，也即$-m\leq \theta \leq \pi - m$，另一条件$\theta \in [0, \pi]$，综合上述两个条件就可以算出\theta的范围，即$0 \leq \theta \leq \pi -m$。因为$\cos$函数单调递减，所以$\cos\theta \geq \cos(\pi-m)$。只要$\theta$满足上述限制条件，就能保证$\cos$函数单调。

2、对于$\theta$不满足限制条件该怎么处理呢，也即$\cos \theta < \cos(\pi - m)$？

### 1.3  CosFace

$loss$公式：
$$
L = -\frac{1}{N} \sum_{i=1}^N log\frac{e^{s.(\cos(\theta_j,i)-m)}}{e^{s.(\cos(\theta_j,i)-m)}+\sum_{j=1,j\neq y_i}^ne^{s.\cos(\theta_j,i)}}
$$
其中：
$$
\begin{align*}
W &= \frac{W}{||W^*||}  \\
x &= \frac{x}{||x^*||}  \\
\cos(\theta_j, i) &= W_j^T x_i
\end{align*}
$$
