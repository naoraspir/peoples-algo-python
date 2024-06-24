import os

class Config:
    # to notify http url upon finishinng placed in this ENV variable   
    NOTIFICATION_HTTP_API = 'https://us-central1-peoples-software.cloudfunctions.net/EmailSender'
    TIMEOUT = 60*60*24 # 24 hours
    FIRESTORE_DATABASE = "peoples-prod"
    FIRESTORE_COLLECTION = "contacts" # collection name
    FIRESTORE_STATUS_FIELD = "notified" # field name
    FIRESTORE_EMAIL_FIELD = "email" # field name
    FIRESTORE_STATUS_NOTIFIED=True # status value
    FIRESTORE_STATUS_NOT_NOTIFIED=False # status value    
    

