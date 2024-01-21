import os

import pandas as pd
import numpy as np

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Gmail API Setup
# https://console.cloud.google.com/apis/credentials/consent?project=smart-bonus-340819
test_user = "carkodw@gmail.com"
# https://developers.google.com/gmail/api/auth/scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
# generate credentials at https://console.cloud.google.com/apis/credentials?project=smart-bonus-340819


def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret_314002233314-giv5f2img0n3dcq60qi445kd3te04cpd.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def get_emails(service, user_id='me', max_results=10):
    response = service.users().messages().list(userId=user_id, maxResults=max_results).execute()
    messages = []
    for msg in response['messages']:
        msg_data = service.users().messages().get(userId=user_id, id=msg['id']).execute()
        messages.append(msg_data['snippet'])
    return messages


def main():

    # Load your labeled dataset (replace 'job_application_dataset.csv' with your actual dataset)
    # The dataset should have two columns: 'text' (email content) and 'label' (category)
    df = pd.read_csv('job_application_dataset.csv')
    count = len(df.index)

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
    emails = get_emails(gmail_service, max_results=count)

    # Classify emails
    email_texts = emails  # Replace this line with actual extraction of email content
    email_features = vectorizer.transform(email_texts)
    predictions = classifier.predict(email_features)
    actual = np.array(df['label'])

    # Print predictions
    for email, prediction in zip(emails, predictions):
        print(f'Email: {email}')
        print(f'Predicted Category: {prediction}\n')

    # Evaluate the performance of the classifier on the test set
    accuracy = accuracy_score(actual, predictions)
    conf_matrix = confusion_matrix(actual, predictions)
    classification_rep = classification_report(actual, predictions)

    print(f'Accuracy: {accuracy:.2f}')
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print('\nClassification Report:')
    print(classification_rep)

if __name__ == "__main__":
  main()
