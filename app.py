from flask import Flask, render_template, request, redirect, url_for, session 
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
import random
from nltk import classify,NaiveBayesClassifier,FreqDist
from nltk.tokenize import word_tokenize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/review',methods=['GET','POST'])
def review():
    return render_template('sub.html')

@app.route('/review2',methods=['GET','POST'])
def review2():
    v1=''
    result=''
    if request.method=="POST":
        v1=request.form['review']
    
    # print(v1)
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

    def lemmatize_sentence(tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []
        for word, tag in pos_tag(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_sentence

    def remove_noise(tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    stop_words = stopwords.words('english')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    def get_all_words(cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token
    
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freqdistpos = FreqDist(all_pos_words)

    def get_tweets_for_model(cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)
    
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    posData = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

    negData = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]
    
    dataset = posData + negData

    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]
    classifier = NaiveBayesClassifier.train(train_data)
    print("input")
    textInput = v1
    custom_tokens = remove_noise(word_tokenize(textInput))
    print(classifier.classify(dict([token, True] for token in custom_tokens)))
    result=classifier.classify(dict([token, True] for token in custom_tokens))
    return render_template('sub2.html',result=result)
    if __name__=="__main__":
        app.run(debug=True)