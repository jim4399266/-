
# Homework-week7
## 前三题必做

## 1. MATCHSUM 里面 pearl-summary是什么？为什么要找到pearl-summary？
- 由于之前的抽取式摘要模型都是基于句子级（Sentence-level）提取的，对所有句子逐个打分，最后取topn的句子为摘要，而MATCHSUM是利用BertSum先抽取m个句子组成候选集，再从中选出n个句子组成摘要级组合（Summary-level），利用摘要级组合整体与标准摘要进行计算得出Summary-level Score。
- Sentence-level Score是将摘要级组合中的每个句子与标准摘要计算得分在平均。
pearl-summary是指在句子级打分较低，摘要级打分高的组合。
- 由于论文中表示大部分最佳摘要不是由得分最高的句子组成的，所以需要找到pearl-summary；并且pearl-summary在所有最佳摘要中的比例是一个用来表征数据集的属性，这将影响我们对摘要提取器的选择。


## 2. 知识蒸馏里参数 T（temperature）的意义？
- 首先来说softmax函数，它对于由于是基于指数的运算。对于输入的数据来说，方差越大，那么softmax的结果就越陡，方差越小，softmax的结果就越平缓。参数T就是用来调节输入数据的方差，当T>1时，softmax的输入越平缓，当T<1时，softmax输出越陡峭。
- 其次来看蒸馏的过程，蒸馏存在两部分的loss，第一部分是student model和 one-hot 标签(hard target)的loss，第二部分是student model 和 teacher model 的 tempered softmax(soft target)的loss。
而teacher model 的 softmax是将原始输入除以T再经过softmax得出的，这里的T一般都大于1，是为了让teacher model 的 softmax更平缓一些，让student model学到更多teacher model的知识。


## 3. TAPT（任务自适应预训练）是如何操作的？
- 由于DAPT使用了大量领域内的无标签数据进行继续的预训练，虽然在领域内的下游任务效果提升了，但是对于领域外的下游任务效果大部分相较于原始预训练模型还下降了，因此，在大多数情况下，不考虑领域相关性而直接直接暴露于更多数据的持续预训练对最终任务可能是有害的。
- 而TAPT选用了任务相关的无标注数据集继续进行预训练（一小部分，大部分还是用于task的训练），任务数据集可以看作相关领域数据的一个子集。相比于DAPT，TAPT使用的是预训练语料要少得多，但是与特定任务相关的语料要多得多。
- TAPT效果要比DAPT差一些。

## 附加思考题（可做可不做）：
从模型优化的角度，在推理阶段，如何更改MATCHSUM的孪生网络结构？
- MATCHSUM在推理阶段由两个带权值的bert和一个余弦相似层组成。
需要利用BertSum抽取出m个候选句，再选择n句进行组合；用bert分别计算文章和候选组合的向量，进行相似度计算，选择最好的组合作为推理答案。
- 唉没有啥好想法。
