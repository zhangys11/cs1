from threading import Timer
import webbrowser
import sys
import json
import os

if __package__:
    from ..metrics import *
else:
    DIR = os.path.dirname(os.path.dirname(__file__))  # cs1 dir
    if DIR not in sys.path:
        sys.path.append(DIR)
    from metrics import *

from flask import Flask, render_template, request
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # limit to 5MB

# routes


@app.route("/", methods=['GET', 'POST'])
def index():
	return render_template("home.html")


@app.route("/about")  # GET
def about_page():
    return "Created by Dr. Zhang (oo@zju.edu.cn)"


@app.route("/submit", methods=['GET', 'POST'])
def receiver():  # Receiver(float k, string timestamp, string xs, string XArray, string XAxisMeaning, string XAxisUnit)
    r = ''
    if request.method == 'POST':
        '''
                k = request.form["k"]
        timestamp = request.form["timestamp"]
        xs = request.form["xs"]
        XArray = request.form["XArray"]
        XAxisMeaning = request.form["XAxisMeaning"]
        XAxisUnit = request.form["XAxisUnit"]
        '''
        return render_template('receiver.html', ViewBag=request.form) # {'message': 'success', 'html': r}  # render_template("home.html", use_sample = use_sample, d = d, nobs = n, cla_result = r)

'''
The Flask dev server is not designed to be particularly secure, stable, or efficient. 
By default it runs on localhost (127.0.0.1), change it to app.run(host="0.0.0.0") to run on all your machine's IP addresses.
0.0.0.0 is a special value that you can't use in the browser directly, you'll need to navigate to the actual IP address of the machine on the network. You may also need to adjust your firewall to allow external access to the port.
The Flask quickstart docs explain this in the "Externally Visible Server" section:
    If you run the server you will notice that the server is only accessible from your own computer, not from any other in the network. This is the default because in debugging mode a user of the application can execute arbitrary Python code on your computer.
    If you have the debugger disabled or trust the users on your network, you can make the server publicly available simply by adding --host=0.0.0.0 to the command line.
'''



def open_browser():
    webbrowser.open_new('http://localhost:5006/')

if __name__ == '__main__':
    # use netstat -ano|findstr 5005 to check port use
    Timer(3, open_browser).start()
    app.run(host="0.0.0.0", port=5006, debug= False)

