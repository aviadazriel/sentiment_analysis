from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
STOP_WORDS = stopwords.words('english')

class preproccessor:
    def __init__(self, df):
        self.columns = ['sentiment', 'label', 'desc']
        if not all(column in df.columns for column in self.columns):
            print(f'Not all required fields are available: {self.columns}')
            return
        self.df = df

    # This function will be our all-in-one noise removal function
    def __remove_noise(self, tokens):
        chars = re.escape(string.punctuation)
        cleaned_tokens = []
        for token, tag in pos_tag(tokens):
            # Eliminating the token if it is a link
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            # Eliminating the token if it is a mention
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)
            token = re.sub(r'[' + chars + ']', '', token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            cleaned_token = token.lower()

            # Eliminating the token if its length is less than 3, if it is a punctuation or if it is a stopword
            if cleaned_token not in string.punctuation and len(cleaned_token) > 2 and cleaned_token not in STOP_WORDS:
                cleaned_tokens.append(cleaned_token)
        return cleaned_tokens

    def clear_data(self):
        self.df['word_tokenize'] = self.df['desc'].apply(lambda x: word_tokenize(x))
        self.df['clean_desc'] = self.df['word_tokenize'].apply(lambda x: self.__remove_noise(x))
        self.df["len"] = self.df['clean_desc'].apply(lambda x: len(x))
        print(f'Data Removing: {Counter(self.df[self.df["len"] <= 3]["sentiment"])}')
        self.df = self.df[self.df["len"] > 3]
        self.df = self.df.reset_index()
        return self.df