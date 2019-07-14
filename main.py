from utils import *
from keras.initializers import RandomUniform
import scipy.sparse as ss
import random

from skmultilearn.problem_transform import ClassifierChain, LabelPowerset
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.cluster import NetworkXLabelGraphClusterer, LabelCooccurrenceGraphBuilder
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.adapt import MLTSVM, BRkNNaClassifier, BRkNNbClassifier, MLkNN
from sklearn.svm import SVC

KERAS_PARAMS = dict(epochs=10, batch_size=100, verbose=0)

parameters = {
    'classifier': [LabelPowerset(), ClassifierChain()],
    'classifier__classifier': [RandomForestClassifier()],
    'classifier__classifier__n_estimators': [10, 20, 50],
    'clusterer': [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}

num_doc = 500
bg_noise = 0.2
random.seed(0)


def main():
    x, y, sequences, word_counts, vocabulary, vocabulary_inv_list, len_avg, all_label_set_kw, train_ids, embedding_mat, sent_length, tf_map = load_data(
        "reuters")
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    vocab_sz = len(vocabulary_inv)
    total_counts = sum(word_counts[ele] for ele in word_counts)
    total_counts -= word_counts[vocabulary_inv_list[0]]
    background_array = np.zeros(vocab_sz)

    # lm = train_lstm(sequences, 10000, sent_length, "toxic", embedding_matrix=embedding_mat)

    for i in range(1, vocab_sz):
        background_array[i] = word_counts[vocabulary_inv[i]] / total_counts
    pseudo_docs, pseudo_label = bow_pseudodocs(all_label_set_kw, background_array, embedding_mat, vocabulary_inv,
                                               len_avg, num_doc, bg_noise, tf_map, doc_len=len(x[0]))

    # pseudo_docs, pseudo_label = lstm_pseudodocs(all_label_set_kw, len(x[0]), len_avg,
    #                                             sent_length, num_doc, embedding_mat, tf_map,
    #                                             vocabulary_inv, lm, 10000, "reuters")

    X_train, y_train = [x[i] for i in train_ids], [y[i] for i in train_ids]
    # X_train = ss.lil_matrix(X_train)
    # y_train = ss.lil_matrix(y_train)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_real_doc = len(pseudo_docs) / 5
    curr_len = len(train_ids)
    copy_times = int(num_real_doc / curr_len) - 1
    new_docs = X_train
    new_y = y_train
    for _ in range(copy_times):
        new_docs = np.concatenate((new_docs, X_train), axis=0)
        new_y = np.concatenate((new_y, y_train), axis=0)

    X_train = np.concatenate([new_docs, pseudo_docs])
    y_train = np.concatenate([new_y, pseudo_label])

    # X_train = ss.lil_matrix(X_train.tolist())
    # y_train = ss.lil_matrix(y_train.tolist())

    # clf = GridSearchCV(LabelSpacePartitioningClassifier(), parameters, scoring='f1_macro')
    # clf = MLTSVM(c_k=2 ** -1)
    # clf = BRkNNaClassifier(k=3)
    # clf = OneVsRestClassifier(SVC(kernel='linear'))
    # X_test, y_test = [x[i] for i in range(len(x)) if i not in train_ids], [y[i] for i in range(len(x)) if
    #                                                                        i not in train_ids]
    # X_test = ss.lil_matrix(X_test)
    # y_test = ss.lil_matrix(y_test)

    # clf.fit(X_train, y_train)
    filter_sizes = [2, 3, 4, 5]
    # clf = rnn(len(x[0]), len(y[0]), vocab_sz=vocab_sz, embedding_matrix=embedding_mat)
    clf = cnn(len(x[0]), filter_sizes=filter_sizes,
              init=RandomUniform(minval=-0.01, maxval=0.01), n_classes=len(y[0]),
              vocab_sz=vocab_sz, embedding_matrix=embedding_mat,
              hidden_dim=100,
              word_embedding_dim=100, num_filters=20,
              word_trainable=False, act='relu')

    X_test, y_test = np.array([x[i] for i in range(len(x)) if i not in train_ids]), np.array(
        [y[i] for i in range(len(x)) if i not in train_ids])
    clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(X_train, y_train, batch_size=256, epochs=500)
    self_train(clf, x, y, train_ids)
    # res = clf.predict(X_test)
    # y_pred = process_proba(res)
    # calc_score(y_test, y_pred)


if __name__ == '__main__':
    main()
