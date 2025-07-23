import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
texts = ["Win money now!", "Hey friend", "Free gift", "Let's meet", "Claim your prize!", "Study hard", "Click here to win"]
labels = [1, 0, 1, 0, 1, 0, 1]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Streamlit UI
st.title("📩 Spam Message Classifier")
message = st.text_input("Enter your message:")

if st.button("Check"):
    input_data = vectorizer.transform([message])
    prediction = model.predict(input_data)[0]
    result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
    st.success(result)
