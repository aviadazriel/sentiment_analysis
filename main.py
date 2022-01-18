from model_biLstm import biLstmModel
from preproccessor import preproccessor
from word_embedding import word_embedding_provider
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd
import numpy as np

def combine_desc(row):
  if row["DESCRIPTION1"].strip().lower() == row["DESCRIPTION2"].strip().lower():
    return f'{row["REVIEW_TITLE"]} {row["DESCRIPTION1"]}'
  else:
    return f'{row["REVIEW_TITLE"]} {row["DESCRIPTION1"]} {row["DESCRIPTION2"]}'

def label2sentiment(label):
    if label >= 4:
        return "Positive"
    elif label == 3:
        return "Natural"
    else:
        return "Negative"

def sentiment2label(sentiment):
        if sentiment == "Positive":
            return 1
        elif sentiment == "Natural":
            return 2
        else:
            return 0
if __name__ == "__main__":
    df = pd.read_csv('amazon_products.csv')
    df['desc'] = df.apply(lambda row: combine_desc(row), axis=1)
    df['sentiment'] = df['RATING'].apply(lambda label: label2sentiment(label))
    df["label"] = df["sentiment"].apply(sentiment2label)
    p = preproccessor(df.copy())
    df = p.clear_data()
    embedding_provider = word_embedding_provider()
    word_to_index, index_to_word, word_to_vec_map = embedding_provider.read_glove_vecs('glove.6B.50d.txt')
    max_len = max(df["len"])
    print('max_len:', max_len)
    X = embedding_provider.get_glove_word_embedding(word_to_index, index_to_word, word_to_vec_map, df['clean_desc'],                                               max_len)
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    # over sample data -SMOTE
    oversample = SMOTE()
    X_train, Y_train = oversample.fit_resample(X_train, Y_train)
    # summarize distribution
    counter = Counter(Y_train)
    for k, v in counter.items():
        per = v / len(Y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    Y_train = tf.keras.utils.to_categorical(Y_train, 3)
    Y_test = tf.keras.utils.to_categorical(Y_test, 3)
    print(f'train size: {len(X_train)}')
    print(f'test size: {len(X_test)}')

    # build and fit model
    biLstm_Model = biLstmModel(word_to_vec_map, word_to_index, max_len)
    biLstm_Model.fit(X_train, Y_train)

    score = biLstm_Model.evaluate(X_test, Y_test)
    print("Testing Accuracy(%): ", score[1] * 100)

    y_pred = biLstm_Model.predict(X_test)
    y_predicted_labels = np.array([np.argmax(i) for i in y_pred])
    y_test_labels = np.array([np.argmax(i) for i in Y_test])

    cm = confusion_matrix(y_test_labels, y_predicted_labels)
    print(cm)
