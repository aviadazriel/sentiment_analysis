




import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


class biLstmModel():
    def __init__(self, word_to_vec_map, word_to_index, max_len):
        # Defining a sequencial model composed of firstly the embedding layer, than a pair of Bidirectional LSTMs,
        # that finally feed into a softmax layer with 3 units
        self.model = Sequential()
        self.model.add(self.__pretrained_embedding_layer(word_to_vec_map, word_to_index, max_len))
        self.model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=3, activation='softmax'))
        self.model.summary()

        self.METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name="auc")]

        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                           metrics=self.METRICS)
        self.custom_early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=4,
            min_delta=0.001,
            mode='max')

    def fit(self, X_train, Y_train, validation_split=0.25, epochs=20, batch_size=512):
        # Setting a batch size of 64 , 20 epochs, 0.2 validation
        self.model.fit(X_train, Y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size,
                       shuffle=True, callbacks=[self.custom_early_stopping])

    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def plot_acc_loss(self):
        history = self.model.history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Defining a function that will initialize and populate our embedding layer

    def __pretrained_embedding_layer(self, word_to_vec_map, word_to_index, max_len):
        vocab_len = len(word_to_index) + 1
        emb_dim = word_to_vec_map["unk"].shape[0]  # 50

        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, idx in word_to_index.items():
            emb_matrix[idx, :] = word_to_vec_map[word]

        embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, input_shape=(max_len,))
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer


