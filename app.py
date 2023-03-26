from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import subprocess
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    # create a connection to the database
    conn = sqlite3.connect('trail.db')
    
    # read the 'recommendation' and 'template' columns into a dataframe
    df = pd.read_sql('SELECT * FROM trail', conn)
    
    # close the database connection
    conn.close()    
    
    # render the HTML template with the data columns
    return render_template('index.html', data=df)

@app.route('/generate_event')
def generate_event():
    subprocess.run(['python', 'runner.py'])
    return redirect(url_for('index'))

@app.route('/audit_trail')
def audit_trail():
    # get the 'id' parameter from the URL
    record_id = request.args.get('id')
    
    # create a connection to the database
    conn = sqlite3.connect('trail.db')
    
    # read the entire row with the specified 'id' into a dataframe
    df = pd.read_sql(f'SELECT * FROM trail WHERE id={record_id}', conn)
    print(record_id)
    
    # close the database connection
    conn.close()
    
    # pass the dataframe to the HTML template
    return render_template('audit_trail.html', data=df)

if __name__ == '__main__':
    app.run(debug=True)

