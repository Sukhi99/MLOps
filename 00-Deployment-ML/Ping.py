from flask import Flask

app = Flask('Test-Web-Service')

@app.route('/simple_web_service', methods = ['GET'])
def ping():
    print("This is test for FLASK")
    return "Pong", 200 

if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port = 9696)

