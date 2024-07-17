import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://face-recog-f5d59-default-rtdb.firebaseio.com/'
})

ref = db.reference("Students")

data = {
    "1" :
        {
            "name" : "Elon musk",
            "Age": "55",
            "YOE" : 5
        }
}

for key,value in data.items():
    ref.child(key).set(value)