from flask import render_template
from flask import request 
from flaskexample import app
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

@app.route('/', methods=['GET','POST'])
#@app.route('/index')
def index():
    
    return render_template("index.html")

@app.route('/game',  methods=['GET','POST'])
def game_page():
    db_name=''
    db_user=''
    db_password=''
    new_ratio=0
    con=psycopg2.connect(database=db_name,user=db_user,host='localhost',password=db_password)
    output_string=""
    name0=""
    name1=""
    name2=""
    url0=""
    url1=""
    url2=""
    img0=""
    img1=""
    img2=""
    final_genres_list=['2D','4X','Survival','Open World','Adventure','Action','RPG','Indie','Dungeon Crawler','Casual','Strategy',
        'Simulation','Sandbox','Shooter','Horror','Sports','Visual Novel','First-Person',
        'Top-Down','Third Person','God Game','Rhythm',"Beat 'em up",'Survival Horror','Battle Royale',
        'Puzzle','PvP','Rogue-like','Tactics','Turn-Based','Real-Time','MOBA','Building','VR Only',
        'Platformer','Walking Simulator','Stealth','Side Scroller','Education','Metroidvania','Space Sim',
        'Management','Fighting','Hack and Slash','Flight','Exploration','Isometric','Runner',
        "Shoot 'Em Up",'Arcade','Racing','Board Game','Card Game','Software']
    final_themes_list=['Western','Sci-fi','Anime','Realistic','Cyberpunk', 'Historical','Cute', 'Lovecraftian',
                       'Space','Relaxing', 'Magic','Retro','Atmospheric','Fantasy','Science','Steampunk',
                       'Zombies','Futuristic','Minimalist','Funny', 'Post-apocalyptic','Dinosaurs','Medieval',
                       'Tactical','Robots','Military','Naval','Colorful','War','Music']
    final_features_list=['Pixel Graphics','Level Editor','Physics', 'Team-Based','Co-op','Multiplayer',
              'Massively Multiplayer','VR','Difficult','Procedural Generation',
              'Character Customization','Controller','Resource Management','Competitive',
              'Rogue-lite','Story Rich','Fast-Paced','Crafting','Split Screen',
                         'Downloadable Content', 'In-App Purchases']
    final_genres_list=sorted(final_genres_list)
    final_themes_list=sorted(final_themes_list)
    final_features_list=sorted(final_features_list)

    num_genres=len(final_genres_list)
    num_themes=len(final_themes_list)
    num_features=len(final_features_list)
    themes_start=num_genres
    features_start=num_genres+num_themes
    themes_end=features_start
    features_end=features_start+num_features

    labels=final_genres_list+final_themes_list+final_features_list

    apps_df=pd.read_sql_query("SELECT * FROM apps_table",con).drop('index',axis=1).set_index('appid')
    norm_game_vecs_df=pd.read_sql_query("SELECT * FROM gvecs_table",con).drop('index',axis=1).set_index('appid')
    
    training_vecs_df=pd.DataFrame.copy(norm_game_vecs_df)
    col_names=training_vecs_df.columns.tolist()
    for i in col_names:
        training_vecs_df[i+i]=training_vecs_df[i]*training_vecs_df[i]
    training_vecs_df['one']=1
    training_vecs=training_vecs_df.values
    training_labels=apps_df['first_month_num_reviews'].values
    regressor=RandomForestRegressor(n_estimators=128,n_jobs=-1,max_features= 'sqrt',random_state=613)
    regressor.fit(training_vecs,training_labels)    


    one_vec=np.zeros(105)
    form_vars=[]
    form_vars.append(int(request.form['genre1']))
    form_vars.append(int(request.form['genre2']))
    form_vars.append(int(request.form['genre3']))
    form_vars.append(int(request.form['theme1']))
    form_vars.append(int(request.form['theme2']))
    form_vars.append(int(request.form['theme3']))
    form_vars.append(int(request.form['feature1']))
    form_vars.append(int(request.form['feature2']))
    form_vars.append(int(request.form['feature3']))
    
    used_vars=list(set([x for x in form_vars if x!=-1]))
    for x in used_vars:
        one_vec[x]=1
    
    def split_vec_norm(x,idx_start,idx_end):
        if np.linalg.norm(x[idx_start:idx_end])!=0:
            if np.linalg.norm(x[idx_start:idx_end])<0.9999 or np.linalg.norm(x[idx_start:idx_end])>1.0001:
                x[idx_start:idx_end]=x[idx_start:idx_end]/np.linalg.norm(x[idx_start:idx_end])
        return x

    def normalized_vec(x):
        x=split_vec_norm(x,0,num_genres)
        x=split_vec_norm(x,themes_start,themes_end)
        x=split_vec_norm(x,features_start,features_end)
        return x

    def false_normalized_vec(x,true_normal_vec):
        if np.sum(true_normal_vec[:themes_start])!=1:
            x=split_vec_norm(x,0,num_genres)
        if np.sum(true_normal_vec[themes_start:themes_end])!=1:
            x=split_vec_norm(x,themes_start,themes_end)
        if np.sum(true_normal_vec[features_start:])!=1:
            x=split_vec_norm(x,features_start,features_end)
        return x

    def perturbed_vec(x,b,used_vars,amt):
        v=np.copy(x)
        for i in range(0,9):
            if 2**i & b:
                v[used_vars[i]]+=amt
        return v

    p_vecs=[]
    if len(used_vars)<=1:
        error_page=True
    else:
        error_page=False
        true_normal_vec=np.copy(one_vec)
        true_normal_vec=normalized_vec(true_normal_vec)
        false_normal_vec=np.copy(true_normal_vec)
        if np.sum(true_normal_vec[:themes_start])==1:
            false_normal_vec[:themes_start]*=0.8
        if np.sum(true_normal_vec[themes_start:themes_end])==1:
            false_normal_vec[themes_start:themes_end]*=0.8
        if np.sum(true_normal_vec[features_start:])==1:
            false_normal_vec[features_start:]*=0.8
        
        
        bigger=[]
        smaller=[]


        for i in range(2**len(used_vars)-1):
            temp_v=perturbed_vec(false_normal_vec,i,used_vars,0.05)
            p_vecs.append(false_normalized_vec(temp_v,true_normal_vec))
            temp_v=perturbed_vec(false_normal_vec,i,used_vars,0.1)
            p_vecs.append(false_normalized_vec(temp_v,true_normal_vec))
            temp_v=perturbed_vec(false_normal_vec,i,used_vars,0.15)
            p_vecs.append(false_normalized_vec(temp_v,true_normal_vec))
            temp_v=perturbed_vec(false_normal_vec,i,used_vars,0.2)
            p_vecs.append(false_normalized_vec(temp_v,true_normal_vec))
        


        p_df=pd.DataFrame(np.vstack(p_vecs),columns=norm_game_vecs_df.columns)
        col_names=p_df.columns.tolist()
        for i in col_names:
            p_df[i+i]=p_df[i]*p_df[i]
        p_df['one']=1

        predicted=regressor.predict(p_df.values)
        
        max_idx=np.argmax(predicted)
        new_ratio=max(1,int(100*(predicted[max_idx]-predicted[0])/predicted[0]+0.5))
        for x in used_vars:
            if p_vecs[max_idx][x]>false_normal_vec[x]:
                bigger.append(labels[x])
            elif p_vecs[max_idx][x]<false_normal_vec[x]:
                smaller.append(labels[x])

        if len(bigger)==0 and len(smaller)==0:
            output_string="</h3><h2>We couldn't find a way to suggest a focus for your idea! Your idea may be unusual, or your input may be too vague to generate a response (try adding more tags).</h2><h3>"
        else:
            output_string=""
            if len(bigger)>0:
                output_string+="We suggest <i>emphasizing</i> the following aspect"
                if len(bigger)>1:
                    output_string+="s"
                output_string+=" of your idea at launch to increase its appeal to the Steam userbase:<br><h2>"+bigger[0]
                if len(bigger)>1:
                    for i in range(1,len(bigger)-1):
                        output_string+=", "+bigger[i]
                    for i in range(len(bigger)-1,len(bigger)):
                        output_string+=" and "+bigger[-1]
                output_string+="</h2><h3>"
                if len(smaller)>0:
                    output_string+="<br><br><br>The following aspect"
                    if len(smaller)>1:
                        output_string+="s"
                    output_string+=" of your idea <i>may be less important</i>:<br><h2>"+smaller[0]
                    if len(smaller)>1:
                        for i in range(1,len(smaller)-1):
                            output_string+=", "+smaller[i]
                        for i in range(len(smaller)-1,len(smaller)):
                            output_string+=" and "+smaller[-1]
                    output_string+="</h2><h3>"
            elif len(smaller)>0:
                output_string+="We suggest <i>de</i>-emphasizing the following aspect"
                if len(smaller)>1:
                    output_string+="s"
                output_string+=" of your idea at launch to increase its appeal to the Steam userbase:<br><h2>"+smaller[0]
                if len(smaller)>1:
                    for i in range(1,len(smaller)-1):
                        output_string+=", "+smaller[i]
                    for i in range(len(smaller)-1,len(smaller)):
                        output_string+=" and "+smaller[-1]
                output_string+="</h2><h3>"
        apps_df['log']=apps_df['first_month_num_reviews'].apply(lambda x: max(1,np.log(x)))
        apps_df['sqrtlog']=apps_df['log'].apply(lambda x: max(1,np.sqrt(x)))
        dot_df=norm_game_vecs_df.dot(true_normal_vec)
        used_appids=[]
        tempapplist=dot_df.multiply(apps_df['sqrtlog']).sort_values(ascending = False).index.tolist()
        for x in tempapplist:
            if not apps_df.loc[x]['content_warning']:
                used_appids.append(x)
                break
        tempapplist=dot_df.sort_values(ascending = False).index.tolist()
        for x in tempapplist:
            if x not in used_appids and not apps_df.loc[x]['content_warning']:
                used_appids.append(x)
                break
        tempapplist=dot_df.sort_values(ascending = False).index.tolist()
        for x in tempapplist:
            if x not in used_appids and not apps_df.loc[x]['content_warning']:
                used_appids.append(x)
                break
        name0=apps_df.loc[used_appids[0]]['name']
        name1=apps_df.loc[used_appids[1]]['name']
        name2=apps_df.loc[used_appids[2]]['name']
        url0='https://store.steampowered.com/app/'+str(used_appids[0])
        url1='https://store.steampowered.com/app/'+str(used_appids[1])
        url2='https://store.steampowered.com/app/'+str(used_appids[2])
        img0='https://steamcdn-a.akamaihd.net/steam/apps/'+str(used_appids[0])+'/header.jpg'
        img1='https://steamcdn-a.akamaihd.net/steam/apps/'+str(used_appids[1])+'/header.jpg'
        img2='https://steamcdn-a.akamaihd.net/steam/apps/'+str(used_appids[2])+'/header.jpg'
    
    return render_template("game.html",output_str=output_string,error_page=error_page,name0=name0,name1=name1,name2=name2,url0=url0,url1=url1,url2=url2,
                                img0=img0,img1=img1,img2=img2,ratio=  new_ratio  )


