from flask import render_template
from flask import request 
from flaskexample import app
from flaskexample.a_Model import ModelIt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import pickle
import pandas as pd
import psycopg2
import requests
import json
import bs4

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'qpc0'
password = 'temp'     # change this
host     = 'localhost'
#port     = '5432'            # default port that postgres listens on
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = password) #add your Postgres password here

@app.route('/', methods=['GET','POST'])
#@app.route('/index')
def index():
    
    return render_template("index.html")

@app.route('/game',  methods=['GET','POST'])
def game_page():
    split_str= str(request.form['storeurl']).split('/')
    appid=0
    if 'app' in split_str:
        appid=split_str[split_str.index('app')+1]
    try:
        page = requests.get('https://store.steampowered.com/app/'+appid).text
    except:
        pass
    soup = bs4.BeautifulSoup(page, 'html.parser')
    temp_str=soup.find("meta",  property="og:title")['content']
    temp_idx=temp_str.rfind(" on Steam")
    if temp_idx>1:
        temp_str=temp_str[:temp_idx]
    mydivs = soup.findAll("div", {"class": "user_reviews_summary_row"})
    div_strs=[]
    for div in mydivs:
        try:
            div_strs.append(div["data-tooltip-text"])
        except:
            pass
    header_im_url='https://steamcdn-a.akamaihd.net/steam/apps/'+appid+'/header.jpg'
    with open('20190124_model','rb') as file:
        regressor = pickle.load(file)
    x=0
    if appid in os.listdir('20190124_ea_vecs'):
        found_record=True
        with open(os.path.join('20190124_ea_vecs',appid),'rb') as file:
            v = pickle.load(file)
        x=regressor.predict(v)[0]
    else:
        found_record=False
    
    return render_template("game.html",apptitle=temp_str, divstrs=div_strs, header_url=header_im_url,
       foundrecord=found_record, prediction=float(int(x*1000+0.5))/10)

@app.route('/db')
def birth_page():
    sql_query = """                                                             
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
;                                                                               
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    print(query_results[:10])
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('birth_month')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  print(query)
  query_results=pd.read_sql_query(query,con)
  print(query_results)
  births = []
  for i in range(0,query_results.shape[0]):
      births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
      the_result = ModelIt(patient,births)
  return render_template("output.html", births = births, the_result = the_result)

