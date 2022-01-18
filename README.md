# sentiment_analysis
Deep learning models (NLP) to implement sentiment analysis

In this task I built a sentiment classifier using BiLSTM network based on data collected from Amazon Data.

The sentiment of the review was determined according to the rating of the review, when:

• 1-2 : Negative

• 3: Natural

• 4-5 : Positive

Each review has a title and 2 descriptions (description 1 and description 2). To include all the 
information, the title and descriptions were combined into a comprehensive description (if 
description 1 was the same as description 2, only description 1 was taken).

<b>Preprocessing</b>

Data cleaning:

• Punctuation - Punctuations are often unnecessary as it doesn’t add value or meaning to the 
NLP model

• Remove stop words

• Lemmatizing - the process of reducing a word to its root form. The main purpose is to 
reduce variations of the same word, thereby reducing the corpus of words we include in the 
model.

• Remove URL

If after data cleaning process, a review contains at most 3 words, the review is removed.
Features

Word Embedding - I used a word2vec pre-training model by glove to make word representation of 
the reviews.

<b>Data Split</b>

After clearing the text and creating the embedding, I divided the data into train and test for the 
model training (division 80/20).

Imbalanced data

Because there is an imbalance in the data, I created synthetic data so that the model could be 
trained in a balanced way, without tilting to a particular label, using the SMOTE (overload) algorithm.

<b>Model</b>

The model consists of four layers:
1. Embedding layer
2. BiLSTM
3. BiLSTM (dropout 0.5)
4. Output layer (softmax)

Model definition:

Batch size : 512 ,Epoch = 20 , Optimizer : Adam (learning rate 0.005)

<b>Result</b>

<table>
<thead>
<td>Train</td>
<td>Test</td>
<td>Metrics</td>
</thead>
<tbody>
<tr>
<td>0.9</td>
<td>0.88</td>
<td>Accuracy</td>
</tr>
<tr>
<td>0.94</td>
<td>0.93</td>
<td>AUC</td>
</tr>
</tbody>
</table>

<b>Evaluate</b>

The model's results are quite good, although it was difficult for him to identify the reviews with the 
neutral sentiment. The model also suffered from a slight overfeeding for the positive sentiment that 
was the cause of the data imbalance.

I think there are two logical explanations:

1. The words in the responses between the positive responses 
and the neutral responses were very similar

2. The model does not consider the context of the word in the 
text


