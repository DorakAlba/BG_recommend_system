import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for,session
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import numpy as np
from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
import joblib
import pandas as pd

app = Flask(__name__)

preds_df = joblib.load('preds_df.pkl')
df_games = joblib.load('df_games.pkl')
bgg_r2 = joblib.load('bgg_r2.pkl')
df_ratings = joblib.load('df_ratings.pkl')



@app.route('/')
def hello_world():
    print ('hi')
    return '<h1>Hello, this my recommendation system for board games'

@app.route('/show_image')
def show_image():
    return '<img src ="static/pic4039881.jpg",alt="flower">'


app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        g = form.name.data
        session["g"] = g
        return redirect('/success')
        return(str(form.name))
    return render_template('submit.html', form=form)
@app.route('/success')
def success():
    g = session.get("g", None)
    Your_id = bgg_r2[bgg_r2['user'] == '%s' % g]["userIdx"].iloc[0]
    sorted_user_predictions = preds_df.iloc[Your_id - 1].sort_values(ascending=False)
    user_data = df_ratings[df_ratings.userIdx == (Your_id)]
    user_full = (user_data.merge(df_games, how='left', left_on='gameId', right_on='gameId').
                 sort_values(['Bayes average'], ascending=False)
                 )
    recommendations = (df_games[~df_games['gameId'].isin(user_full['gameId'])]).merge(
        pd.DataFrame(sorted_user_predictions).reset_index(), how='left', left_on='gameId',
        right_on='index').rename(columns={Your_id - 1: 'Predictions'}).sort_values('Predictions',ascending=False).iloc[:10]
    url = []
    for line in recommendations['name']:
        url.append('https://boardgamegeek.com/' + bgg_r2[bgg_r2['name'] == line]["URL"].iloc[0][0])
    recommendations['URL'] = url
    return render_template('simple.html', tables=[recommendations.to_html(classes='data')], titles=recommendations.columns.values)
