import pandas as pd
import re
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt

#Loaded dataset
df = pd.read_csv(r"C:\Users\r.lavanya\Downloads\Sample_Support_Emails_Dataset.csv")

#Filter emails
keywords = ["support", "query", "request", "help"]
df = df[df['subject'].str.contains('|'.join(keywords), case=False, na=False)]

#Sentiment Analysis
def get_sentiment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"
df["Sentiment"] = df["body"].apply(get_sentiment)

#Priority
urgent_words = ["immediately", "urgent", "critical", "cannot access", "asap"]
def get_priority(text):
    return "Urgent" if any(word in str(text).lower() for word in urgent_words) else "Not Urgent"

df["Priority"] = df["body"].apply(get_priority)

#Contact Info
def extract_contacts(text):
    emails = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', str(text))
    phones = re.findall(r'\b\d{10}\b', str(text))
    return ", ".join(emails + phones)

df["Contacts"] = df["body"].apply(extract_contacts)

#Auto-Response Draft
def generate_reply(subject, sentiment):
    if sentiment == "Negative":
        return f"Hi, we understand your concern about '{subject}'. We're looking into it urgently."
    else:
        return f"Hi, thanks for reaching out about '{subject}'. We'll get back to you shortly."

df["AI Draft Reply"] = df.apply(lambda row: generate_reply(row["subject"], row["Sentiment"]), axis=1)

#Dashboard
st.title("📩 Support Email Dashboard")

#KPIs
st.subheader("Quick Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Emails", len(df))
col2.metric("Urgent Emails", sum(df["Priority"]=="Urgent"))
col3.metric("Negative Sentiments", sum(df["Sentiment"]=="Negative"))

#Sentiment Chart
st.subheader("Sentiment Distribution")
sentiment_counts = df["Sentiment"].value_counts()
fig1, ax1 = plt.subplots()
sentiment_counts.plot(kind="bar", ax=ax1)
st.pyplot(fig1)

#Priority Chart
st.subheader("Priority Distribution")
priority_counts = df["Priority"].value_counts()
fig2, ax2 = plt.subplots()
priority_counts.plot(kind="bar", ax=ax2)
st.pyplot(fig2)

#Full Table
st.subheader("Filtered Support Emails")
st.dataframe(df[["sender", "subject", "Sentiment", "Priority", "Contacts", "AI Draft Reply"]])
