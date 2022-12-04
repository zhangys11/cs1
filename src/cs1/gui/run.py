from threading import Timer
import webbrowser
import sys
import json
import os


if __package__:
    from ..cs import Recovery, MeasurementMatrix
else:
    # pip install cs1
    from cs1.cs import Recovery, MeasurementMatrix
    '''
    DIR = os.path.dirname(os.path.dirname(__file__))  # cs1 dir
    if DIR not in sys.path:
        sys.path.append(DIR)
    from metrics import *
    from cs import *
    '''

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
    return render_template('tutorial.html')

@app.route("/receiver", methods=['GET', 'POST'])
def receiver():  # Receiver(float k, string timestamp, string xs, string XArray, string XAxisMeaning, string XAxisUnit)
    
    if request.method == 'POST':

        dic = {}

        dic["k"] = request.form["k"] # distance between means, respect to std, i.e. (mu2 - mu1) / std, or how many stds is the difference.
        dic["timestamp"] = request.form["timestamp"] # number of observations / samples			
        dic["xs"] = request.form["xs"]
        dic["XArray"] = request.form["XArray"]
        dic["XAxisMeaning"] = request.form["XAxisMeaning"]
        dic["XAxisUnit"] = request.form["XAxisUnit"]

        return render_template('receiver.html', ViewBag=dic) # {'message': 'success', 'html': r}  # render_template("home.html", use_sample = use_sample, d = d, nobs = n, cla_result = r)

    else:
        dic = {'k': '0.1', 'timestamp': '11/17/2022, 10:04:44 PM', 'xs': '[2330.0,2331.0,2332.0,2333.0,2334.0,2335.0,2336.0,2337.0,2338.0,2339.0]', 'XAxisMeaning': 'Wave Number', 'XAxisUnit': 'cm-1'} # for test only
        
        return render_template('receiver.html', ViewBag=dic)

@app.route("/reconstruct", methods=['GET', 'POST'])
def reconstruct():  # Receiver(float k, string timestamp, string xs, string XArray, string XAxisMeaning, string XAxisUnit)
    
    if request.method == 'POST':

        phi = request.form["phi"] 
        phi = list(map(int, phi.split(",")))# convert to list
        xs = request.form["xs"] 
        xs = list(map(float, xs[1:-1].split(",")))# convert to list or np array
        psi = request.form["psi"]
        n = int( request.form["n"] )


        #     A = csmtx(n, phi, t = psi)
        #     xr, z = lasso_cs(A, xs, 0.005, psi, silent = True)
        A = MeasurementMatrix(n, phi, t = psi)
        z,xr = Recovery (A, xs, t = psi, PSI = None, solver = 'LASSO', \
        L1 = 0.005, display = False, verbose = True) # solver = OMP

        d = {}
        d['xr'] = xr.flatten().tolist() #[ '%.2f' % elem for elem in xr.tolist() ]
        d['z'] = z.flatten().tolist()

        return d
    
    # 
    # otherwise, you may create a local file for data interchange
    fn = os.path.dirname(os.path.realpath(__file__)) + "/" + str(uuid.uuid4()) + ".json"
    with open(fn, 'w') as f:
        json.dump(d, f)

    #print(os.getcwd())
    print(fn)
    #print(json.dumps(d))

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

