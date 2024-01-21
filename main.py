import os
import asyncio
import pandas as pd
import numpy as np
from email.utils import parseaddr

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# NLP
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer


load_dotenv()

# Gmail API Setup
# https://console.cloud.google.com/apis/credentials/consent?project=smart-bonus-340819
# https://developers.google.com/gmail/api/auth/scopes
# generate credentials at https://console.cloud.google.com/apis/credentials?project=smart-bonus-340819
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

recruitment_vocab = ["interview", "softwar", "applic", "appli"]
second_stage_phrases = ["senior", "invite", "technical interview", "calendar"]

def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(os.client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def get_emails(service, user_id='me', max_results=10):
    response = service.users().messages().list(userId=user_id, maxResults=max_results).execute()
    messages = []
    orgs = []
    addresses = []
    for msg in response['messages']:
        msg_data = service.users().messages().get(userId=user_id, id=msg['id']).execute()
        messages.append(msg_data['snippet'])
        sender = [header["value"] for header in msg_data["payload"]["headers"] if header["name"] == "From"][0]
        email = parseaddr(sender)[1]
        addresses.append(email)
        org = " ".join(email.split("@")[1].split(".")[:-1]).title()
        orgs.append(org)

    return messages, addresses, orgs

def setup_ntlk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download("maxent_ne_chunker")
    nltk.download('words')


def language_processing(content):
    # Remove zero-width characters
    content = content.strip(' ͏')
    original_content = content
    text = content.encode()
    text = text.decode()
    tokens = sent_tokenize(text)
    words_in_quote = word_tokenize(text)
    eng_stop_words = set(stopwords.words("english"))
    filtered_list = [word for word in words_in_quote if (word.casefold() not in eng_stop_words) and (word.isalnum())]
    stemmer = EnglishStemmer()
    lemmatizer = WordNetLemmatizer()
    # filtered_list = [stemmer.stem(word) for word in filtered_list]
    # filtered_list = [lemmatizer.lemmatize(word) for word in filtered_list]
    recruitment_tags = nltk.pos_tag(filtered_list)

    # e.g. Your application was successful, you applied successfully, your application was received
    # PRP -> RB -> VBD -> NN
    application_phrase_structure = "NP: {<PRP>?<RB|NN><NN>*}"
    chunk_parser = nltk.RegexpParser(application_phrase_structure)
    tree = chunk_parser.parse(recruitment_tags)
    classification = None
    
    for word in filtered_list:
        if stemmer.stem(word) in recruitment_vocab:
            classification = "Applied"
            if stemmer.stem(word) in second_stage_phrases:
                classification = "In progress"
                return filtered_list, classification
    
        if lemmatizer.lemmatize(word) in recruitment_vocab:
            classification = "Applied"
            if lemmatizer.lemmatize(word) in second_stage_phrases:
                classification = "In progress"
                return filtered_list, classification

    return filtered_list, classification, original_content


async def run_classification():
    setup_ntlk()
    # Load your labeled dataset (replace 'job_application_dataset.csv' with your actual dataset)
    # The dataset should have two columns: 'text' (email content) and 'label' (category)
    df = pd.read_csv('job_application_dataset.csv')
    # count = len(df.index)
    count = 300

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        np.array(df['text']), np.array(df['label']), test_size=0.3, random_state=42
    )

    # Convert the text data into numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(train_features, train_labels)

    # Gmail API
    gmail_service = get_gmail_service()
    emails, addresses, orgs = get_emails(gmail_service, max_results=count)

    # Classify emails
    email_texts = emails
    with open("recruitment_emails_train.csv", "w") as content:
        for i, e in enumerate(emails):
            # label = classifier.predict(vectorizer.transform([e]))[0]
            filtered_list, classification, original_content = language_processing(e)
            # if detected:
            #     print(f"This is recruitment application email {e}")
            #     label = None
            if classification:
                content.write(f"{original_content}, {classification}, {orgs[i]}\n")

    # vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
    # X = vect.fit_transform(email_texts)

    # email_features = vectorizer.transform(email_texts)
    # predictions = classifier.predict(email_features)
    # actual = np.array(df['label'])

    # # Print predictions
    # for email, prediction in zip(emails, predictions):
    #     print(f'Email: {email}')
    #     print(f'Predicted Category: {prediction}\n')

    # # Evaluate the performance of the classifier on the test set
    # accuracy = accuracy_score(actual, predictions)
    # conf_matrix = confusion_matrix(actual, predictions)
    # classification_rep = classification_report(actual, predictions)

    # print(f'Accuracy: {accuracy:.2f}')
    # print('\nConfusion Matrix:')
    # print(conf_matrix)
    # print('\nClassification Report:')
    # print(classification_rep)


async def main():
    await asyncio.gather(
        asyncio.create_task(run_classification()),
    )
    

if __name__ == "__main__":
  asyncio.run(main())
