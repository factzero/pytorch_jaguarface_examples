## 1.retinaFace设计思路

### 1.1 预备知识

[](https://zhuanlan.zhihu.com/p/42159963)

[](https://blog.csdn.net/zijin0802034/article/details/77685438)

[](http://caffecn.cn/?/question/160)

[](https://www.jianshu.com/p/2c0995908cc3)

#### 1.1.1 边框回归

![](img\img2.jpg)

对于窗口一般使用四维向量$(x,y,w,h)$ 来表示，分别表示窗口的中心点坐标和宽高。上图红色的框 P 代表原始的Proposal, 绿色的框 G 代表目标的 Ground Truth，我们的目标是寻找一种关系使得输入原始的窗口 P 经过映射得到一个跟真实窗口 G 更接近的回归窗口$\hat{G}$。

黄色框T代表真实框(Ground Truth)，白色P代表边界框(bounding box)，红色框G代表P经过边框回归后的预测框，所谓边框回归也就是指边界框P经过调整成为预测框G的过程。这里假设每个边框的位置为$(x,y,w,h)$，分别表示边界框的中心坐标以及宽高，P，$\hat{G}$，G三个框分别表示为：$P = (P_x, P_y, P_w, P_h)$，$\hat{G} = (\hat{G}_x, \hat{G}_y, \hat{G}_w, \hat{G}_h)$，$G = (G_x, G_y, G_w, G_h)$，边框回归也就是要找到一个函数使得$f(P) = \hat{G}$

  即：给定$P = (P_x, P_y, P_w, P_h)$，寻找一种映射$f$，使用$f(P)=\hat{G}$，且$\hat{G} \approx G$

让P尺度扩张再往左移动就可以变成G，即尺度缩放和平移操作。

(1)尺度缩放$(S_w, S_h)$：  $S_w = exp(d_w(P)),  S_h = exp(d_h(P))$
$$
\begin{align*}
\hat{G}_w &= P_wexp(d_w(P)) \\
\hat{G}_h &= P_hexp(d_h(P))
\end{align*}
$$
(2)平移$(\Delta x, \Delta y)$：  $\Delta x = P_w d_x(P),  \Delta y = P_h d_y(P)$
$$
\begin{align*}
\hat{G}_x &= P_w d_x(P) + P_x \\
\hat{G}_y &= P_h d_y(P) + P_y
\end{align*}
$$
边框回归就是学习$d_x(P), d_y(P), d_w(P), d_h(P)$这四个变换，下一步就是设计算法那得到这四个映射。线性回归就是给定输入的特征向量 X, 学习一组参数 W, 使得经过线性回归后的值跟真实值 Y(Ground Truth)非常接近，即$Y \approx WX$ 。那么 Bounding-box 中我们的输入以及输出分别是什么呢？

Input：$RegionProposal(P_x, P_y, P_w, P_h)$，特征向量$W_*$，Ground Truth$t_* = (t_x, t_y, t_w, t_h)$

Output：$d_x(P), d_y(P), d_w(P), d_h(P)$

$d_*(P)$是得到的预测值，$t_*$是对应的真实值，目标就是两者差距最小，损失函数为：
$$
Loss = \sum_i(t_*^i - d_*^i(P))
$$
可利用梯度下降法求解。

对应的目标真实值平移量$(t_x, t_y)$和尺度缩放$(t_w, t_h)$，计算公式如下：
$$
\begin{align*}
t_x &= (G_x - P_x) / P_w \\
t_y &= (G_y - P_y) / P_h \\
t_w &= log(G_w/P_w)  \\
t_h &= log(G_h/P_h)  \\
\end{align*}
$$

### 1.2 先验框(prior box)

​       [](https://www.cnblogs.com/pacino12134/p/10353959.html)

​        借鉴anchor的理念，为feature map的每个单元设置长宽比相同的先验框(人脸长宽基本相同，而其它目标检测领域需要设置不同的长宽比)。借用SSD相关图。每个单元会设置多个先验框，其尺度和长宽比存在差异，如图所示，可以看到每个单元使用了4个不同的先验框，图片中猫和狗分别采用最适合它们形状的先验框来进行训练，后面会详细讲解训练过程中的先验框匹配原则。

![](img\img1.jpg)

对于每个单元的每个先验框，其都输出一套独立的检测值，对应一个边界框，主要分为两个部分。

第一部分是各个类别的置信度，背景也当做了一个特殊的类别，如果检测目标共有c个类别，则需要预测c+1个置信度值，其中第一个置信度指的是不含目标或者属于背景的评分。后面当我们说c个类别置信度时，请记住里面包含背景那个特殊的类别，即真实的检测类别只有c−1个。在预测过程中，置信度最高的那个类别就是边界框所属的类别，特别地，当第一个置信度值最高时，表示边界框中并不包含目标。
第二部分是边界框的location包含4个值$(cx, cy, w, h)$，分别表示边界框的中心坐标以及宽高。假设先验框位置为$p = (p^{cx}, p^{cy}, p^{w}, p^{h})$，其对应的边界框为$b = (b^{cx}, b^{cy}, b^{w}, b^{h})$，那么边界框的预测值$l = (l^{cx}, l^{cy}, l^{w}, l^{h})$

计算公式如下：
$$
\begin{align*}
l^{cx} &= (b^{cx} - p^{cx})/p^{w}, l^{cy} = (b^{cy} - p^{cy})/p^{h}\\
l^{w} &= log(b^{w} / p^{w}), l^{h} = log(b^{h} / p^{h})
\end{align*}
$$
习惯上，称上述过程为边界框的编码(encode)，预测时需要反向这个过程，即解码(decode)，从预测值$l$到边界框的真实位置$b$

计算公式如下：
$$
\begin{align*}
b^{cx} &= p^{w}l^{cx} + p^{cx}, b^{cy} = p^{h}l^{cy} + p^{cy}\\
b^{w} &= p^{w}exp(l^{w}), b^{h} = p^{h}exp(l^{h})
\end{align*}
$$
在训练中使用了一些trick，那就是设置variance超参数来调整检测值，此时边界框的编码和解码过程要更新。

编码更新如下：
$$
\begin{align*}
l^{cx} &= (b^{cx} - p^{cx})/(p^{w}*variance[0]), l^{cy} = (b^{cy} - p^{cy})/(p^{h}*variance[1])\\
l^{w} &= log(b^{w} / p^{w})/variance[2], l^{h} = log(b^{h} / p^{h})/variance[3]
\end{align*}
$$
解码更新如下：
$$
\begin{align*}
b^{cx} &= p^{w}(variance[0]*l^{cx}) + p^{cx}, b^{cy} = p^{h}(variance[1]*l^{cy}) + p^{cy}\\
b^{w} &= p^{w}exp(variance[2]*l^{w}), b^{h} = p^{h}exp(variance[3]*l^{h})
\end{align*}
$$
对于一个大小$m×n$的特征图，共有$mn$个单元，每个单元设置的先验框数目记为$k$，那么每个单元共需要$(c+4)k$个预测值，所有的单元共需要$(c+4)kmn$个预测值。



## 2.训练过程

### 2.1先验框匹配

​        在训练过程中，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。先验框与ground truth的匹配有两个原则：

第一个原则：对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。通常称与ground truth匹配的先验框为正样本，反之，若一个先验框没有与任何ground truth进行匹配，那么该先验框只能与背景匹配，就是负样本。一个图片中ground truth是非常少的， 而先验框却很多，很多先验框会是负样本，正负样本极其不平衡。

第二个原则：对于剩余的未匹配先验框，若某个ground truth的IOU大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配。

​        尽管一个ground truth可以与多个先验框匹配，但是ground truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3。

### 2.2损失函数

