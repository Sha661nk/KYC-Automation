import flask
import json
from flask import request,jsonify
from flask_cors import CORS, cross_origin
import KYC_Automation_Driver

app = flask.Flask(__name__)
cors = CORS(app)

print("Initializing Data")

@app.route('/validation', methods=['POST'])
@cross_origin()
def Validation():
    fullname = request.get_json()["fullname"]
    gender = request.get_json()["gender"]
    dependentname = request.get_json()["dependentname"]
    adhaar = request.get_json()["adhaar"]
    email = request.get_json()["email"]
    dob = request.get_json()["dob"]
    account = request.get_json()["account"]
    contact = request.get_json()["contact"]
    address = request.get_json()["address"]
    applicationno = request.get_json()["applicationno"]
    address = address.strip()
    address = address[0:len(address)-5]
    #EXTRACT PINCODE
    pincode = address[len(address)-7:len(address)]
    print("Pincode for this validation process is:" + pincode)
    KYC_Automation_Driver.Executor(fullname, dob, gender.upper(), str(adhaar), dependentname, address, int(pincode), 'front.jpg', 'back.jpg', 'pic.jpg', 'crop_emblem_2.jpg', 'crop_goi.jpg', applicationno)
    return 'Success'

@app.route('/validation',methods=['GET'])
@cross_origin()
def api_all():
	return 'success'

app.run()
