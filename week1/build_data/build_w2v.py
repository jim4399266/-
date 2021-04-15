from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os

from utils import config
from utils.multi_proc_utils import cores


def build_w2v(save_model=False):
    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(config.merger_seg_path),
                        size=config.embedding_dim,
                        sg=1,
                        workers=cores,
                        iter=config.wv_train_epochs,
                        window=5,
                        min_count=5)
    if save_model and config.save_wv_model_path:
        if not os.path.exists(config.save_wv_model_path):
            os.makedirs(os.path.dirname(config.save_wv_model_path))
        wv_model.save(config.save_wv_model_path)
        print('Model has been saved!')
    return wv_model

if __name__ == '__main__':
    model = build_w2v(True)

