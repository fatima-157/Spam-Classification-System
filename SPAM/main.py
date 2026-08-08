from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def create_spam_classifier(X_train, y_train):
    """
    Creates and trains a Multinomial Naive Bayes pipeline for spam detection.
    """
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

def classify_message(model, message):
    """
    Classifies a text message as spam or ham.
    """
    prediction = model.predict([message])
    return "Spam" if prediction[0] == 1 else "Ham"
