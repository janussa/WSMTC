import pandas as pd
import scipy.sparse as ss
import nltk
import random
import itertools
import re
import os
import pickle
import json
from tqdm import tqdm
from keras.models import Model
from keras.layers import *
from keras.layers.merge import Concatenate
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
from nltk import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from gensim.models import word2vec
from summa import keywords
from keras.callbacks import ModelCheckpoint
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder, NetworkXLabelGraphClusterer
from spherecluster import VonMisesFisherMixture, sample_vMF
from time import time
from multiprocessing import Pool
from keras.optimizers import SGD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')
ps = PorterStemmer()
wl = WordNetLemmatizer()
cv = CountVectorizer()
random.seed(0)

# categories = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
num_labels = len(categories)


def vec_2_lbs(vec):
    return [categories[j] for j in range(num_labels) if vec[j] == 1]


def rnn(input_shape, n_classes, word_trainable=False, vocab_sz=None,
        embedding_matrix=None, word_embedding_dim=100, hidden_dim=100):
    x = Input(shape=(input_shape,), dtype='int32')
    if embedding_matrix is not None:
        z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,),
                      weights=[embedding_matrix], trainable=word_trainable)(x)
    else:
        z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,), trainable=word_trainable)(x)
    z = GRU(100)(z)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = Dense(hidden_dim, activation="relu")(z)
    y = Dense(n_classes, activation="sigmoid")(z)
    return Model(inputs=x, outputs=y)


def cnn(input_shape, n_classes, filter_sizes, init, num_filters=20, word_trainable=False,
        vocab_sz=None,
        embedding_matrix=None, word_embedding_dim=100, hidden_dim=100, act="relu"):
    x = Input(shape=(input_shape,), dtype='int32')
    if embedding_matrix is not None:
        z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,),
                      weights=[embedding_matrix], trainable=word_trainable)(x)
    else:
        z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,), trainable=word_trainable)(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation=act,
                             strides=1,
                             kernel_initializer=init)(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = BatchNormalization()(z)
    # z = Dropout(0.3)(z)
    z = Dense(hidden_dim, activation="relu")(z)
    y = Dense(n_classes, activation="sigmoid")(z)
    return Model(inputs=x, outputs=y)


def calc_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "length not matched"
    TP, FP, TN, FN = np.zeros(num_labels), np.zeros(num_labels), np.zeros(num_labels), np.zeros(num_labels)
    for j in range(num_labels):
        print(f"collect info for category {categories[j]}")
        for i in range(len(y_true)):
            if y_true[i][j] == 1 and y_pred[i][j] == 1:
                TP[j] += 1
            elif y_true[i][j] == 0 and y_pred[i][j] == 1:
                FP[j] += 1
            elif y_true[i][j] == 0 and y_pred[i][j] == 0:
                TN[j] += 1
            elif y_true[i][j] == 1 and y_pred[i][j] == 0:
                FN[j] += 1
    macro_f1 = sum((TP * 2) / (TP * 2 + FN + FP)) / num_labels
    avg_TP = sum(TP) / num_labels
    avg_FP = sum(FP) / num_labels
    avg_FN = sum(FN) / num_labels
    micro_f1 = (avg_TP * 2) / (avg_TP * 2 + avg_FN + avg_FP)
    print(f"macro F1:{macro_f1}, micro F1:{micro_f1}")
    return macro_f1, micro_f1


def doc_filter(X, y, min_len=5):
    X_new, y_new = [], []
    for doc, label in tqdm(zip(X, y)):
        temp = []
        sents = nltk.sent_tokenize(doc)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            if len(words) >= min_len:
                temp.extend(words)
        if len(temp) >= min_len:
            X_new.append(" ".join(temp))
            y_new.append(label)
    return X_new, y_new


def sample_ids(y, dataset):
    if os.path.exists(f"{dataset}/train_ids.json"):
        with open(f"{dataset}/train_ids.json") as f:
            node_ids = json.load(f)
        f.close()
        all_ids = [x for node in node_ids for x in node_ids[node]]
        id_set = set(all_ids)
        return [x for x in id_set], node_ids
    y_df = pd.DataFrame(y)
    id_set = set()
    node_ids = {}
    for i in range(num_labels):
        ids = y_df[y_df[i] == 1].index.to_list()
        ids = random.sample(ids, 3)
        node_ids[i] = ids
        for idx in ids:
            id_set.add(idx)
    with open(f"{dataset}/train_ids.json", "w") as f:
        json.dump(node_ids, f)
    f.close()
    return [x for x in id_set], node_ids


def pad_docs(sentences, pad_len=None, padding_word="<PAD/>"):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for sentence in sentences:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, common_words):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    trim_vocabulary = {}
    for i, x in enumerate(vocabulary_inv):
        if i < common_words:
            trim_vocabulary[x] = i
        else:
            trim_vocabulary[x] = common_words
    vocab_inv = {}
    for i in range(len(vocabulary_inv)):
        vocab_inv[i] = vocabulary_inv[i]
    embedding_model = word2vec.Word2Vec(sentences, workers=15, sg=1)
    embedding_model.init_sims(replace=True)
    embedding_weights = {key: embedding_model[word] if word in embedding_model else np.random.uniform(-0.25, 0.25,
                                                                                                      embedding_model.vector_size)
                         for key, word in vocab_inv.items()}
    embedding_mat = np.array([np.array(embedding_weights[word]) for word in vocab_inv])
    return word_counts, vocabulary, vocabulary_inv, trim_vocabulary, embedding_mat


def build_input_data(sentences, vocabulary):
    x = [[vocabulary[word] for word in sentence] for sentence in sentences]
    return x


def build_sequence(flat_data, vocabulary, truncate_len):
    flat_data = build_input_data(flat_data, vocabulary)
    sequences = []
    for seq in flat_data:
        for i in range(1, len(seq)):
            sequence = seq[:i + 1]
            sequences.append(sequence)
    sequences = pad_sequences(sequences, maxlen=truncate_len, padding="pre")
    print("Sequences shape: {}".format(sequences.shape))
    return sequences


def extract_keywords_tfidf(label_set_ids, data, num_keywords=10):
    # data = [" ".join(line) for line in data]
    tfidf = TfidfVectorizer(norm="l2", sublinear_tf=True, max_df=0.2, stop_words="english",
                            token_pattern=r"(?u)\b\w[\w-]*\w\b", max_features=10000)
    all_idx = []
    for node in label_set_ids:
        all_idx += label_set_ids[node]
    all_text = [data[idx] for idx in all_idx]
    x_all = tfidf.fit_transform(all_text)
    vocab_dict = tfidf.vocabulary_
    vocab_inv_dict = {v: k for k, v in vocab_dict.items()}
    cum_cnt = 0
    node_keywords = {}
    for node in label_set_ids:
        x_node = x_all[cum_cnt:cum_cnt + len(label_set_ids[node])].todense()
        cum_cnt += len(label_set_ids[node])
        class_vec = np.average(x_node, axis=0)
        class_vec = np.ravel(class_vec)
        sort_idx = np.argsort(class_vec)[::-1]
        keyword = []
        j = 0
        k = 0
        while j < num_keywords:
            keyword.append(vocab_inv_dict[sort_idx[k]])
            j += 1
            k += 1
        print(f"{node}:{keyword}")
        node_keywords[node] = keyword
    return node_keywords


# def extract_keywords_lda(vocab, data, node_ids, embedding_weights):
#     node_keywords = {}
#     for i in range(num_labels):
#         # class_vec = embedding_weights[vocab[categories[i]]]
#         X = [data[i] for i in node_ids[i]]
#         X4lda = [nltk.word_tokenize(x) for x in X]
#         X4lda = [[w for w in s if w not in [",", "."]] for s in X4lda]
#         dictionary = gensim.corpora.Dictionary(X4lda)
#         bow_corpus = [dictionary.doc2bow(doc) for doc in X4lda]
#         lda_model = gensim.models.LdaMulticore(bow_corpus,
#                                                num_topics=3,
#                                                id2word=dictionary,
#                                                passes=10,
#                                                workers=2)
#         res = lda_model.print_topics()
#         keywords = [x[1] for x in res]
#         keywords = [re.sub(r"[*+\"]", " ", x) for x in keywords]
#         keywords = [x.split() for x in keywords]
#         keywords = [[w for w in s if re.match(r"[a-z]", w[0])] for s in keywords]
#         print(f"{categories[i]} : {keywords}")
#         kw_vec = [[embedding_weights[vocab[w]] for w in s] for s in keywords]


def extract_keywords(data, train_ids, y, mode="tfidf"):
    y_train = [y[i] for i in train_ids]
    y_train_lil = ss.lil_matrix(y_train)
    partition = clusterer.fit_predict(None, y_train_lil)
    groups = [x for x in partition]
    combinations = []
    for g in groups:
        length = len(g)
        if length < 2:
            continue
        for i in range(length - 1):
            for j in range(i + 1, length):
                temp = list()
                temp.append(categories[g[i]])
                temp.append(categories[g[j]])
                combinations.append("+".join(temp))
    label_train = []
    for yt in y_train:
        temp = []
        for i, e in enumerate(yt):
            if e == 1:
                temp.append(categories[i])
        label_train.append("+".join(temp))
    label_set_ids = {}
    for i, idx in enumerate(train_ids):
        if label_train[i] not in label_set_ids:
            label_set_ids[label_train[i]] = []
        label_set_ids[label_train[i]].append(idx)
    label_set_kw = {}
    if mode == "tr":
        for labels in label_set_ids:
            docs = [data[i] for i in label_set_ids[labels]]
            text = " ".join(docs)
            kw = keywords.keywords(text)
            kw = kw.split('\n')
            kw = [w for s in kw for w in s.split()]
            kw = [w for w in kw if w.__len__() > 2]
            label_set_kw[labels] = kw
    else:
        label_set_kw = extract_keywords_tfidf(label_set_ids, data)
    all_combinations = set()
    for c in combinations:
        all_combinations.add(c)
    for l in label_set_ids:
        all_combinations.add(l)
    all_label_set_kw = {}
    for labels in all_combinations:
        kw_set = set()
        if labels in label_set_kw:
            for kw in label_set_kw[labels]:
                kw_set.add(kw)
        subs = labels.split("+")
        if len(subs) > 1:
            for sub in subs:
                if sub in label_set_kw:
                    for kw in label_set_kw[sub]:
                        kw_set.add(kw)
        kw_list = [w for w in kw_set]
        if len(kw_list) > 3:
            all_label_set_kw[labels] = kw_list
    for l in all_label_set_kw:
        print(f"{l} : {all_label_set_kw[l]}")
    return all_label_set_kw


def preprocess_doc(X, stopwords):
    X_words = [x.split() for x in X]
    new_X = []
    for xw in X_words:
        nx = [w for w in xw if w not in stopwords and re.match(r"[a-z,.]", w[0])]
        nx = [w if len(w) <= 20 else w[:20] for w in nx]
        new_X.append(nx)
    X = [" ".join(x) for x in new_X]
    word_list = [nltk.word_tokenize(x) for x in tqdm(X)]
    word_list = [[w for w in s if w not in stopwords] for s in word_list]
    word_list = [[ps.stem(w) for w in s] for s in word_list]
    word_list = [[wl.lemmatize(w) for w in s] for s in word_list]
    X = [" ".join(w) for w in word_list]
    return X


def lstm_lm(input_shape, word_embedding_dim, vocab_sz, hidden_dim, embedding_matrix):
    x = Input(shape=(input_shape,), name='input')
    z = Embedding(vocab_sz, word_embedding_dim, input_length=input_shape, weights=[embedding_matrix], trainable=False)(
        x)
    z = LSTM(hidden_dim, activation='relu', return_sequences=True)(z)
    z = LSTM(hidden_dim, activation='relu')(z)
    z = Dense(vocab_sz, activation='softmax')(z)
    model = Model(inputs=x, outputs=z)
    model.summary()
    return Model(inputs=x, outputs=z)


def train_lstm(sequences, vocab_sz, truncate_len, save_path, word_embedding_dim=100, hidden_dim=100,
               embedding_matrix=None):
    if embedding_matrix is not None:
        trim_embedding = np.zeros((vocab_sz + 1, word_embedding_dim))
        trim_embedding[:-1, :] = embedding_matrix[:vocab_sz, :]
        trim_embedding[-1, :] = np.average(embedding_matrix[vocab_sz:, :], axis=0)
    else:
        trim_embedding = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = save_path + '/model-final.h5'
    model = lstm_lm(truncate_len - 1, word_embedding_dim, vocab_sz + 1, hidden_dim, trim_embedding)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists(model_name):
        print(f'Loading model {model_name}...')
        model.load_weights(model_name)
        return model
    x, y = sequences[:, :-1], sequences[:, -1]
    checkpointer = ModelCheckpoint(filepath=save_path + '/model-{epoch:02d}.h5', save_weights_only=True, period=1)
    model.fit(x, y, batch_size=256, epochs=25, verbose=1, callbacks=[checkpointer])
    model.save_weights(model_name)
    return model


def calc_tf(words, counts, docs):
    docs = [doc.split() for doc in docs]
    res = []
    for i, w in tqdm(enumerate(words)):
        bgdocs = [doc for doc in docs if w in doc]
        denominator = np.sum([len(doc) for doc in bgdocs])
        tf = counts[i] / denominator
        res.append(tf)
    # res = res / np.sum(res)
    return res


def load_data(dataset):
    X = pd.read_csv(f"{dataset}/docs.csv", header=None)
    X = X.values.tolist()
    X = [x[0] for x in X]
    if os.path.exists(f"{dataset}/stopwords"):
        with open(f"{dataset}/stopwords") as f:
            lines = f.readlines()
            stopwords = [l.split('\n')[0] for l in lines]
        f.close()
    else:
        stopwords = sw.words('english')
    X = preprocess_doc(X, stopwords)
    y = pd.read_csv(f"{dataset}/labels.csv", header=None)
    y = y.values.tolist()
    X, y = doc_filter(X, y)

    cv_fit = cv.fit_transform(X)
    counts = cv_fit.toarray().sum(axis=0)
    words = cv.get_feature_names()
    tf = calc_tf(words, counts, X)

    train_ids, node_ids = sample_ids(y, dataset)
    lens_word = [len(nltk.word_tokenize(x)) for x in tqdm(X)]
    lens_word.sort()
    max_len = lens_word[int(len(lens_word) * 0.9)]
    data = [s.split(" ") for s in X]
    trun_data = [s[:max_len] for s in data]
    tmp_list = [len(doc) for doc in data]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print("\n### Dataset statistics - Documents: ###")
    print("Document max length: {} (words)".format(len_max))
    print("Document average length: {} (words)".format(len_avg))
    print("Document length std: {} (words)".format(len_std))

    print("Defined maximum document length: {} (words)".format(max_len))
    print("Fraction of truncated documents: {}".format(sum(tmp > max_len for tmp in tmp_list) / len(tmp_list)))

    sequences_padded = pad_docs(trun_data, pad_len=max_len)
    word_counts, vocabulary, vocabulary_inv, trim_vocabulary, embedding_mat = build_vocab(sequences_padded, 10000)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
    x = build_input_data(sequences_padded, vocabulary)
    x = np.array(x)
    y = np.array(y)
    # Prepare sentences for training LSTM language model
    trun_data = [" ".join(doc) for doc in trun_data]
    flat_data = [nltk.sent_tokenize(doc) for doc in trun_data]
    flat_data = [sent for doc in flat_data for sent in doc]
    flat_data = [sent for sent in flat_data if len(sent.split(" ")) > 5]
    tmp_list = [len(sent.split(" ")) for sent in flat_data]
    max_sent_len = max(tmp_list)
    avg_sent_len = np.average(tmp_list)
    std_sent_len = np.std(tmp_list)
    truncate_sent_len = min(int(avg_sent_len + 3 * std_sent_len), max_sent_len)
    print("\n### Dataset statistics - Sentences: ###")
    print("Sentence max length: {} (words)".format(max_sent_len))
    print("Sentence average length: {} (words)".format(avg_sent_len))
    print("Sentence average std: {} (words)".format(std_sent_len))
    print("Defined maximum sentence length: {} (words)".format(truncate_sent_len))
    print(
        "Fraction of truncated sentences: {}".format(sum(tmp > truncate_sent_len for tmp in tmp_list) / len(tmp_list)))
    flat_data = [s.split(" ") for s in flat_data]
    sequences = build_sequence(flat_data, trim_vocabulary, truncate_sent_len)

    # num_keywords = 10
    # extract_keywords_tfidf(vocabulary, num_keywords, data, node_ids)
    # extract_keywords_lda(vocabulary, trun_data, node_ids, embedding_weights)
    all_label_set_kw = extract_keywords(trun_data, train_ids, y)
    for label_set in all_label_set_kw:
        filtered_kw = [kw for kw in all_label_set_kw[label_set] if kw in vocabulary]
        all_label_set_kw[label_set] = filtered_kw

    tf_map = {}
    for i in range(len(words)):
        if words[i] in vocabulary:
            tf_map[words[i]] = tf[i]

    return x, y, sequences, word_counts, vocabulary, vocabulary_inv, len_avg, all_label_set_kw, train_ids, embedding_mat, truncate_sent_len, tf_map


def sample_mix_vmf(center, kappa, weight, num_doc):
    distrib_idx = np.random.choice(range(len(center)), num_doc, p=weight)
    samples = []
    for idx in distrib_idx:
        samples.append(sample_vMF(center[idx], kappa[idx], 1))
    samples = np.array(samples)
    samples = np.reshape(samples, (num_doc, -1))
    return samples


def label_expansion(all_label_set_kw, vocabulary_inv, embedding_mat):
    print("Retrieving top-t nearest words...")
    vocab_dict = {v: k for k, v in vocabulary_inv.items()}
    prob_sup_array = []
    current_szes = []
    for label_set in all_label_set_kw:
        current_sz = len(all_label_set_kw[label_set])
        current_szes.append(current_sz)
        prob_sup_array.append([1 / current_sz] * current_sz)
    centers = []
    kappas = []
    weights = []
    for label_set in all_label_set_kw:
        num_child = len(label_set.split("+"))
        kw = np.array([vocab_dict[w] for w in all_label_set_kw[label_set]])
        expanded_mat = embedding_mat[np.asarray(kw, dtype="int32")]
        vmf_soft = VonMisesFisherMixture(n_clusters=num_child, n_jobs=15, random_state=0)
        vmf_soft.fit(expanded_mat)
        center = vmf_soft.cluster_centers_
        kappa = vmf_soft.concentrations_
        weight = vmf_soft.weights_
        print("weight: {}".format(weight))
        print("kappa: {}".format(kappa))
        centers.append(center)
        kappas.append(kappa)
        weights.append(weight)

    print("Finished vMF distribution fitting.")
    return centers, kappas, weights


def trans(label_set):
    label_list = label_set.split("+")
    label = np.zeros(len(categories), dtype=np.int32)
    for i in range(len(categories)):
        if categories[i] in label_list:
            label[i] = 1
    return label


def calc_similarity(doc, tf_map):
    cv_fit = cv.fit_transform(doc)
    counts = cv_fit.toarray().sum(axis=0)
    tf = counts / counts.sum()
    words = cv.get_feature_names()
    vec_cps, vec_doc = [], []
    for i in range(len(words)):
        if words[i] in tf_map:
            vec_cps.append(tf_map[words[i]])
            vec_doc.append(tf[i])
    vec_cps = np.array(vec_cps).reshape(1, -1)
    vec_doc = np.array(vec_doc).reshape(1, -1)
    return cosine_similarity(vec_doc, vec_cps).tolist()[0][0]


def bow_pseudodocs(all_label_set_kw, background_array, embedding_mat, vocabulary_inv, len_avg, num_doc, bg_noise,
                   tf_map, doc_len, total_num=50):
    n_classes = len(all_label_set_kw)
    label_set_list = [x for x in all_label_set_kw]
    centers, kappas, weights = label_expansion(all_label_set_kw, vocabulary_inv, embedding_mat)
    background_vec = bg_noise * background_array
    selected_docs = []
    label = []
    for i in range(n_classes):
        # docs = np.zeros((num_doc, doc_len), dtype="int32")

        docs = []
        similarities = []
        docs_len = len_avg * np.ones(num_doc)
        center = centers[i]
        kappa = kappas[i]
        weight = weights[i]
        discourses = sample_mix_vmf(center, kappa, weight, num_doc)
        for j in range(num_doc):
            doc = np.zeros(doc_len, dtype="int32")
            discourse = discourses[j]
            prob_vec = np.dot(embedding_mat, discourse)
            prob_vec = np.exp(prob_vec)
            sorted_idx = np.argsort(-prob_vec)
            delete_idx = sorted_idx[total_num:]
            prob_vec[delete_idx] = 0
            prob_vec /= np.sum(prob_vec)
            prob_vec *= 1 - bg_noise
            prob_vec += background_vec
            cut_doc_len = int(docs_len[j])
            # docs[j][:doc_len] = np.random.choice(len(prob_vec), size=doc_len, p=prob_vec)
            doc[:cut_doc_len] = np.random.choice(len(prob_vec), size=cut_doc_len, p=prob_vec)
            similarity = calc_similarity([vocabulary_inv[x] for x in doc], tf_map)
            docs.append(doc.tolist())
            similarities.append(similarity)
            # label[j] = trans(label_set_list[i])
        similarities = np.array(similarities)
        ids = similarities.argsort()[-8:]
        for idx in ids:
            selected_docs.append(docs[idx])
            label.append(trans(label_set_list[i]))
    docs_df = pd.DataFrame(selected_docs)
    selected_docs = docs_df.values
    label = np.array(label)
    return selected_docs, label


def lstm_pseudodocs(all_label_set_kw, sequence_length, len_avg, sent_length, num_doc, embedding_mat,
                    tf_map, vocabulary_inv, lm, common_words, save_dir=None):
    n_classes = len(all_label_set_kw)
    label_set_list = [x for x in all_label_set_kw]
    centers, kappas, weights = label_expansion(all_label_set_kw, vocabulary_inv, embedding_mat)
    seed_words = []
    for i in range(n_classes):
        center = centers[i]
        kappa = kappas[i]
        weight = weights[i]
        discourses = sample_mix_vmf(center, kappa, weight, num_doc)
        prob_mat = np.dot(discourses, embedding_mat.transpose())
        seeds = np.argmax(prob_mat, axis=1)
        seed_words.append(seeds)
    doc_len = int(len_avg)
    # docs = np.zeros((num_doc * n_classes, sequence_length), dtype='int32')
    # label = np.zeros((num_doc * n_classes, len(categories)))
    selected_docs = []
    label = []
    for i in range(n_classes):
        # seeds = np.reshape(seeds, (num_doc, num_sent))
        docs = []
        similarities = []
        docs_class = gen_with_seeds(label_set_list[i], lm, seed_words[i], doc_len, sent_length, common_words,
                                    vocabulary_inv, save_dir=save_dir)
        for j in range(num_doc):
            doc = np.zeros(sequence_length, dtype="int32")
            doc[:doc_len] = docs_class[j]
            similarity = calc_similarity([vocabulary_inv[x] for x in doc], tf_map)
            docs.append(doc.tolist())
            similarities.append(similarity)
            # label[i * num_doc + j] = trans(label_set_list[i])
        similarities = np.array(similarities)
        ids = similarities.argsort()[-8:]
        for idx in ids:
            selected_docs.append(docs[idx])
            label.append(trans(label_set_list[i]))

    return selected_docs, label


def gen_with_seeds(class_name, lm, seeds, doc_len, sent_length, common_words, vocabulary_inv, save_dir=None):
    t0 = time()
    pool = Pool(10)
    doc_len = int(doc_len)

    sent_cnt = 0
    print(f'Pseudodocs generation for class {class_name}...')

    cur_seq = [[] for _ in range(len(seeds))]
    for i in range(doc_len):
        if i % sent_length == 0:
            pred_real = [seed for seed in seeds]
            pred_trim = [min(seed, common_words) for seed in seeds]
            temp_seq = [[] for _ in range(len(seeds))]
            sent_cnt += 1
        else:
            padded_seq = pad_sequences(temp_seq, maxlen=sent_length - 1, padding='pre')
            pred = lm.predict(padded_seq, verbose=0)
            args = [(common_words, len(vocabulary_inv), ele) for ele in pred]
            res = pool.starmap(gen_next, args)
            pred_real = [ele[0] for ele in res]
            pred_trim = [ele[1] for ele in res]
            assert len(pred_real) == len(cur_seq)
        for j in range(len(cur_seq)):
            cur_seq[j].append(pred_real[j])
            temp_seq[j].append(pred_trim[j])

    cur_seq = np.array(cur_seq)
    print(f'Pseudodocs generation time: {time() - t0:.2f}s')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, f'{class_name}_pseudo_docs.txt'), 'w')
    for seq in cur_seq:
        f.write(" ".join([vocabulary_inv[ele] for ele in seq]) + '\n')
    f.close()
    with open(os.path.join(save_dir, f'{class_name}_pseudo_docs.pkl'), 'wb') as f:
        pickle.dump(cur_seq, f, protocol=4)
    return cur_seq


def gen_next(common_words, total_words, pred):
    select = np.random.choice(common_words + 1, p=pred)
    pred_trim = select
    if select == common_words:
        pred_real = np.random.choice(range(common_words, total_words))
    else:
        pred_real = select
    return pred_real, pred_trim


def process_proba(y_mat):
    for y_line in y_mat:
        order = y_line.argsort()
        y_line[order[-1]] = 1
    y_pred = [[0 if ele < 0.2 else 1 for ele in col] for col in y_mat]
    return np.array(y_pred)


def get_confident_ids(output):
    out_max = np.array([x.max() for x in output])
    # ids = []
    # for i in range(len(out_max)):
    #     if out_max[i] > 0.95:
    #         ids.append(i)
    rank = out_max.argsort()
    ids = rank[-10:]
    return np.array(ids)


def self_train(clf, x, y, train_ids, max_iter=5000):
    # clf.compile(optimizer=SGD(lr=5e-4, momentum=0.9, decay=1e-6), loss='kld')
    X_test = [x[i] for i in range(len(x)) if i not in train_ids]
    y_test = [y[i] for i in range(len(y)) if i not in train_ids]
    X_train = [x[i] for i in train_ids]
    y_train = [y[i] for i in train_ids]
    y_pred, y_pred_last = [], []
    for it in range(max_iter):
        print(f"\n{it} / {max_iter}")
        q = clf.predict(np.array(X_test))
        y_pred = process_proba(q)
        confident_ids = get_confident_ids(q)
        X_con = [X_test[i] for i in confident_ids]
        y_con = [y_pred[i] for i in confident_ids]
        X_mix = X_train + X_con
        y_mix = y_train + y_con
        if it == 0:
            y_pred_last = np.copy(y_pred)
        else:
            change_idx = []
            for i in range(len(y_pred)):
                if not np.array_equal(y_pred[i], y_pred_last[i]):
                    change_idx.append(i)
            y_pred_last = np.copy(y_pred)
            delta_label = len(change_idx)
            print(f"Fraction of documents with label changes: {np.round(delta_label / y_pred.shape[0] * 100, 3)} %")
            if delta_label / y_pred.shape[0] < 0.5 / 100:
                print(f"\nFraction: {np.round(delta_label / y_pred.shape[0] * 100, 3)} % < tol: 0.5 %")
                print("Reached tolerance threshold. Self-training terminated.")
                break
        clf.fit(np.array(X_mix), np.array(y_mix))
    # y_pred = [y_pred[i] for i in range(len(x)) if i not in train_ids]
    calc_score(y_test, y_pred)
