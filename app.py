from flask import Flask, render_template, redirect, url_for, request
import event_gen
import pandas as pd
import subprocess
import sqlite3
from datetime import datetime
import sys

from dotenv import load_dotenv

def setup():
    load_dotenv()
    print('keys loaded')
setup()

app = Flask(__name__)
unsampled_rows = None

def get_random_trail():
    global unsampled_rows

    if unsampled_rows is None or unsampled_rows.empty:
        conn = sqlite3.connect('trail.db')
        query = "SELECT * FROM trail"
        df_trail = pd.read_sql_query(query, conn)
        conn.close()
        unsampled_rows = df_trail.copy()

    random_trail = unsampled_rows.sample(n=1)
    unsampled_rows.drop(random_trail.index, inplace=True)
    random_trail['request_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return random_trail

def save_event(row):
    conn = sqlite3.connect('trail.db')
    row.to_sql('event', conn, if_exists='append', index=False)
    conn.close()

@app.route('/')
def index():
    # create a connection to the database
    conn = sqlite3.connect('trail.db')
    
    # read the 'recommendation' and 'template' columns into a dataframe
    df = pd.read_sql('SELECT * FROM event', conn)
    
    # close the database connection
    conn.close()    
    
    # render the HTML template with the data columns
    return render_template('index.html', data=df)

@app.route('/generate_event')
def generate_event():
    #event_gen.run()

    random_trail = get_random_trail()
    save_event(random_trail)

    return redirect(url_for('index'))


@app.route('/audit_trail')
def audit_trail():
    # get the 'id' parameter from the URL
    record_id = request.args.get('id')
    print(record_id)
    
    # create a connection to the database
    conn = sqlite3.connect('trail.db')
    
    # read the entire row with the specified 'id' into a dataframe
    df = pd.read_sql(f'SELECT * FROM event WHERE eventid={record_id}', conn)
    print(df.employee_goals[0])
    
    # close the database connection
    conn.close()
    
    # pass the dataframe to the HTML template
    return render_template('audit_trail.html', data=df)

if __name__ == '__main__':
    app.run(debug=True)

