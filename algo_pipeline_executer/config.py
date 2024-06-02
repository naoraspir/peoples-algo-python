import os

class Config:
    # to notify http url upon finishinng placed in this ENV variable   
    NOTIFICATION_HTTP_API = 'https://us-central1-peoples-software.cloudfunctions.net/EmailSender'
    TIMEOUT = 60*60*24 # 24 hours
