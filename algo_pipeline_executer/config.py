import os

class Config:
    # to notify http url upon finishinng placed in this ENV variable   
    NOTIFICATION_HTTP_API = 'https://us-central1-peoples-software.cloudfunctions.net/EmailSender'
    TIMEOUT = 60*60*24 # 24 hours
    FIRESTORE_DATABASE = "peoples-prod"
    FIRESTORE_COLLECTION = "albums" # collection name
    FIRESTORE_STATUS_FIELD = "workflowStatus" # field name
    FIRESTORE_STATUS_UPLOADED = "uploaded" # status value
    FIRESTORE_STATUS_RUNNING = "runningAlgo" # status value
    FIRESTORE_STATUS_READY = "ready" # status value
    FIRESTORE_EMAIL_FIELD = "email" # field name
    FIRESTORE_PHOTOGRAHPER_FIELD = "photographerName" # field name
    

