import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
import gc
import numpy as np


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def preprocessing_tag_and_stopwords(s, include_adverb=False):
    lemmatizer = WordNetLemmatizer()

    need_tag = [['NN', 'NNS', 'NNP', 'NNPS'], ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                ['JJ', 'JJR', 'JJS']]
    p = ['n', 'v', 'a']  # 'n':名詞，'v':動詞，'a':形容詞

    if include_adverb is True:
        need_tag = [['NN', 'NNS', 'NNP', 'NNPS'], ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                    ['JJ', 'JJR', 'JJS'], ['RB', 'RBR', 'RBS']]
        p = ['n', 'v', 'a', 'r']  # 'n':名詞，'v':動詞，'a':形容詞，'r':副詞

    stopWords = set(stopwords.words('english'))

    word = 'a'
    pos = []
    for i in range(len(s)):
        morph = nltk.word_tokenize(s[i])  #分かち書き
        l = nltk.pos_tag(morph)  #タグ付け
        l_new = []
        for i in l:
            if i[0] not in stopWords:
                for j in range(len(need_tag)):
                    if i[1] in need_tag[j]:
                        word = lemmatizer.lemmatize(i[0], pos=p[j])  #原型にする
                        l_new.append(word+' ')
        pos.append(''.join(l_new))
    return pos


def make_bow(df):
    s = df['snippet']
    pos = preprocessing_tag_and_stopwords(s, include_adverb=False)

    vectorizer = CountVectorizer(max_df=0.5, min_df=0.03)
    X = vectorizer.fit_transform(pos)
    del pos
    gc.collect()
    X_array = X.toarray()
    word_label = np.array(vectorizer.get_feature_names())
    # Tfidf
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    tfidf = tfidf_transformer.fit_transform(X_array)

    features = np.array(tfidf.todense())

    word_num = 200
    word_order = np.argsort(np.sum(features, axis=0))[::-1]
    features = features[:, word_order][:, :word_num]
    word_label = word_label[word_order][:word_num]
    return features, word_label

if __name__ == '__main__':
    from fetch_arxiv import fetch_search_result
    import time

    search_str = input("> ")

    start = time.time()
    df = fetch_search_result(search_str)
    duration = time.time() - start
    print(f"duration: {duration}s")
    # Load file
    features, word_label = make_bow(df)
