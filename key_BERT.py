import pandas as pd
from keybert import KeyBERT
from transformers import pipeline

# Load the dataset
file_path = 'Data/Winter 2024 Scotia DSD Data Set.xlsx'
df = pd.read_excel(file_path)

# Define the topics and their descriptions
topics = {
    '2SV': 'Two-Step Verification, security measure, numerical code, phone, online banking, authentication, identity protection, account security, login security, dual-factor authentication, secure login, verification process, digital security, secure access, account protection, cybersecurity, extra layer of security, secure transactions, user authentication, one-time code, secure login method, multi-factor authentication, online security, enhanced security, protect account',

    'Application_Performance': 'Application Performance, speed, responsiveness, lags, delays, software efficiency, responsive design, fast application, smooth operation, user experience, optimal performance, application speed, real-time performance, minimize delays, enhance responsiveness, improve speed, application optimization, responsive user interface, eliminate lag, boost performance, optimize software, enhance speed, reduce delays, improve application responsiveness, fast loading',

    'Accessibility': 'Accessibility, design, products, devices, services, environments, disabilities, vision, hearing, cognitive abilities, mobility, inclusive design, assistive technologies, accessible products, disability-friendly, inclusive environments, adaptive devices, universal design, accommodating services, diverse abilities, inclusive solutions, accessible technology, user-friendly design, equal access, disability accommodations, accessible devices, inclusive services, cognitive accessibility',

    'Appointment_Booking': 'Appointment Booking, request, advisor,  schedule, booking system, appointment management, meeting scheduling, advisor availability, appointment request, appointment  scheduling tool, book meetings, appointment confirmation, meeting organizer, advisor meetings, appointment scheduling  appointment setting, schedule advisor meeting, book appointments online, appointment coordination, schedule management, booking confirmation',

    'Biometric_Login': 'Biometric Login, logging,  biometric features, fingerprint authentication, facial recognition, biometric security, secure login method, user verification, biometric technology, login authentication, touch ID, face ID, secure biometric authentication, fingerprint login, biometric login system, user identification, biometric user authentication, enhance security, biometric scanning, user-friendly login, secure access',

    'Budgeting': 'Budgeting, feature, tracking, spending, income, spending graphs, financial planning, expense tracking, budget management, income tracking, budgeting feature, financial tracking, spending analysis, budget visualization, expense management, income and expense tracking, budget tracking tool, financial control, budget planning, spending habits, financial graphs, budgetary control, income management, spending categories',

    'Chat': 'Chat, talking, chatbot, live person, messaging, communication, chat interface, customer interaction, live chat, chat support, conversational experience, real-time chat, user engagement, interactive chat, customer communication, chat platform, chat conversation, messaging  virtual chat, chat assistance, conversational interface, chat service, instant messaging',

    'Cheque_Deposit': 'Cheque Deposit, feature, deposit, cheque, phone,  mobile deposit, check deposit, remote deposit, cheque scanning, deposit feature, mobile banking, digital deposit, deposit, check scanning, deposit through  mobile check deposit, remote check deposit, banking, electronic deposit, check submission, deposit convenience, check processing, check deposit',

    'Credit_Score': 'Credit Score, score,  credit score feature, credit history, credit rating, credit report, credit monitoring, creditworthiness, credit assessment, credit score  financial health, credit information, credit score tracking, credit analysis, credit improvement, credit score update, credit scoring system, creditworthiness evaluation, credit report  credit health, credit rating app',

    'Email_Money_Transfer': 'Email Money Transfer, transfer funds, personal accounts, email, Scotiabank  online money transfer, email transfer, digital funds transfer, online banking transfer, money transfer  email payment, electronic money transfer, fund transfer service, email banking, Scotiabank transfer, secure money transfer, email transaction, online transfer service, money transfer platform, email payment solution, Scotiabank online banking',

    'Errors': 'Errors, flaw, fault, design, development, operation, incorrect result, unexpected result, crashes, glitches, software errors, application flaws, coding errors, system faults, development mistakes, operational issues, design flaws, unexpected outcomes, software bugs, application crashes, programming glitches, system errors, coding faults, unexpected results, software defects',

    'Fee': 'Fee, comments, banking fees, impact,  transaction fees, service charges, financial fees, banking costs, user fees, charges, fee structure, cost impact, fee policy, customer financial experience, pricing comments, banking expenses, fees, transaction costs, service fees, fee transparency, financial impact, user cost',

    'Info_Alerts': 'Info Alerts, notifications, inform, important activities, accounts, notification alerts, account updates, informational alerts, notifications, account activity alerts, important events, real-time alerts, account information, notification system, activity notifications, timely alerts, alerts, alert messages, account alerts, crucial updates, information notifications, instant alerts',

    'International_Money_Movement': 'International Money Movement, transferring money internationally, international money transfer services, Western Union, global money transfer, cross-border funds transfer, international remittance, overseas money transfer, currency exchange, international transaction, global banking, foreign funds transfer, Western Union service, international finance, cross-border payment, global remittance, currency transfer, international banking, Western Union  foreign currency exchange',

    'Investments': 'Investments, discussing, investment accounts, GIC, RRSP, retirement savings plan, investment options, financial investments, investment portfolios, GIC accounts, retirement planning, investment strategies, RRSP accounts, wealth management, investment advice, savings plan, diversified investments, investment opportunities, investment products, financial planning, retirement accounts, investment choices, investment planning, investment advisory',

    'Login_and_Logout_Issues': 'Login and Logout Issues, user experience, logging in, logging out, login problems, logout issues, account access, user login, login challenges, sign-in difficulties, logout errors, access issues, login failures, logout challenges, account entry, user authentication, sign-in problems, logout process, application access, login errors, logout difficulties, user sign-in',

    'Quick_Balance': 'Quick Balance, feature, check account balances, signin,  instant balance, account overview, quick access, balance inquiry, feature, real-time balance, quick information, account summary, rapid balance check, immediate balance, balance details,  functionality, speedy balance check, fast account overview, instant information, quick financial snapshot,  balance feature',

    'Request_New_Card': 'Request New Card, feature, request, replacement card, lost, stolen, original card, card replacement, lost card replacement, stolen card request, new card lication, card services, replacement process, damaged card, card renewal, request card online, stolen card replacement, damaged card replacement, new card request, card replacement service, lost card request, card reissue',

    'Rewards': 'Rewards,  rewards feature, SCENE+ points, loyalty program, reward system, customer loyalty, loyalty points,  incentives, reward program, loyalty rewards, customer benefits, loyalty points system,  loyalty, reward redemption, loyalty perks, SCENE+ rewards, customer satisfaction,  loyalty program, reward catalog, loyalty advantages, loyalty bonuses, loyalty features',

    'Save_and_Share_Statements': 'Save and Share Statements, saving, sharing , transaction statements,  statement management, document storage, transaction history, financial statements, document sharing, statement saving,  documents, transaction records, statement organization, document security, financial history, statement storage, record keeping,  statements, document management, statement sharing, financial document'
}


# Initialize KeyBERT model
keybert_model = KeyBERT()

# Initialize BERT-based model from transformers library
bert_model = pipeline('feature-extraction', model='bert-base-uncased', tokenizer='bert-base-uncased')

# Function to extract keywords using KeyBERT
def extract_keywords_keybert(text):
    keywords = keybert_model.extract_keywords(text)
    return [kw[0] for kw in keywords]

# Function to extract keywords using BERT-based model
def extract_keywords_bert(text):
    features = bert_model(text)
    return [word for feature in features for word in feature[0]]

# Function to categorize a review to the correct topic
def categorize_review(review):
    review_keywords_keybert = set(extract_keywords_keybert(review))
    review_keywords_bert = set(extract_keywords_bert(review))

    max_overlap = 0
    selected_topic = None

    for topic, description in topics.items():
        topic_keywords = set(extract_keywords_keybert(description))
        overlap = len(review_keywords_keybert.intersection(topic_keywords))

        if overlap > max_overlap:
            max_overlap = overlap
            selected_topic = topic

    return selected_topic

df['Topic'] = df['Review'].apply(categorize_review)
output_file_path = 'Categorized_Reviews.xlsx'
df.to_excel(output_file_path, index=False)
print(f"Results saved to {output_file_path}")