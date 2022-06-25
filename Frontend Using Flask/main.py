# Importing libraries
from flask import Flask, render_template, request, redirect, send_file, session
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px

model = pickle.load(open('rn_model.pkl', 'rb'))

data = pd.DataFrame()


def processing_df(cbc):
    cbc = cbc.reset_index(drop=True)
    cbc = cbc.drop(["Code", "Sample No.", "Date", 'Conclusion'], axis=1)
    '''cbc['Conclusion'] = cbc['Conclusion'].map({'HM': 0, 'SEP': 1, 'MDA':2, 'AA':3, 
                                           'ITP':4, 'NM':5, 'MF+IDA':6, 'IDA':7, 
                                           'MPY':8, 'MA':9, 'ITP+IDA':10 , 'CML-CP+IDA':11 , 
                                           'CGD':12 , 'HCV':13 , 'ETS':14 , 'Hypersplenism':15 , 
                                           'PV':16 , 'PRCA':17 , 'ACD':18 , 'PCA':19 , 'CDA':20 , 
                                           'GA' :21 , 'Extremly Increased Iron deposition':22})'''
    catag_data = ["WBC Abnormal", "WBC Suspect", "RBC Abnormal", "RBC Suspect",
                  "PLT Abnormal", "PLT Suspect", "IP ABN(WBC)WBC Abn Scattergram"
        , "IP ABN(WBC)Neutropenia", "IP ABN(WBC)Neutrophilia",
                  "IP ABN(WBC)Lymphopenia", "IP ABN(WBC)Lymphocytosis",
                  "IP ABN(WBC)Monocytosis", "IP ABN(WBC)Eosinophilia",
                  "IP ABN(WBC)Basophilia", "IP ABN(WBC)Leukocytopenia",
                  "IP ABN(WBC)Leukocytosis", "IP ABN(WBC)NRBC Present",
                  "IP ABN(WBC)IG Present", "IP ABN(RBC)RBC Abn Distribution",
                  "IP ABN(RBC)Dimorphic Population", "IP ABN(RBC)Anisocytosis",
                  "IP ABN(RBC)Microcytosis", "IP ABN(RBC)Macrocytosis",
                  "IP ABN(RBC)Hypochromia", "IP ABN(RBC)Anemia",
                  "IP ABN(RBC)Erythrocytosis", "IP ABN(RBC)RET Abn Scattergram",
                  "IP ABN(RBC)Reticulocytosis", "IP ABN(PLT)PLT Abn Distribution",
                  "IP ABN(PLT)Thrombocytopenia", "IP ABN(PLT)Thrombocytosis",
                  "IP ABN(PLT)PLT Abn Scattergram", "IP SUS(WBC)Blasts/Abn Lympho?",
                  "IP SUS(WBC)Blasts?", "IP SUS(WBC)Abn Lympho?",
                  "IP SUS(WBC)Left Shift?", "IP SUS(WBC)Atypical Lympho?",
                  "IP SUS(RBC)RBC Agglutination?", "IP SUS(RBC)Turbidity/HGB Interf?",
                  "IP SUS(RBC)Iron Deficiency?", "IP SUS(RBC)HGB Defect?",
                  "IP SUS(RBC)Fragments?", "IP SUS(PLT)PLT Clumps?"]

    cbc[catag_data] = cbc[catag_data].fillna(value=0)
    cbc['Judgment'] = cbc['Judgment'].map({'Positive': 1, 'Negative': 0})

    cbc['Positive(Diff.)'] = cbc['Positive(Diff.)'].map({'Diff.': 1})

    cbc['Positive(Morph.)'] = cbc['Positive(Morph.)'].map({'Morph.': 1})

    cbc['Positive(Count)'] = cbc['Positive(Count)'].map({'Count': 1})

    # replacing all NaN vallues by 0 as per suggested by the domain expert
    cbc[['Judgment', 'Positive(Diff.)', 'Positive(Count)', 'Positive(Morph.)']] = cbc[
        ['Judgment', 'Positive(Diff.)', 'Positive(Count)', 'Positive(Morph.)']].fillna(value=0)

    numeric_data = ['HGB(g/dL)', 'PLT(10^3/uL)', 'Q-Flag(Blasts/Abn Lympho?)', 'Q-Flag(Left Shift?)',
                    'Q-Flag(Atypical Lympho?)', 'Q-Flag(RBC Agglutination?)',
                    'Q-Flag(Turbidity/HGB Interf?)', 'Q-Flag(Iron Deficiency?)', 'Q-Flag(HGB Defect?)',
                    'Q-Flag(Fragments?)', 'Q-Flag(PLT Clumps?)', 'WBC(10^3/uL)', 'RBC(10^6/uL)',
                    'HCT(%)', 'MCV(fL)', 'MCH(pg)', 'MCHC(g/dL)', 'RDW-SD(fL)', 'RDW-CV(%)', 'PDW(fL)',
                    'MPV(fL)', 'P-LCR(%)', 'PCT(%)', 'NRBC#(10^3/uL)', 'NRBC%(/100WBC)',
                    'NEUT#(10^3/uL)', 'LYMPH#(10^3/uL)', 'MONO#(10^3/uL)', 'EO#(10^3/uL)',
                    'BASO#(10^3/uL)', 'NEUT%(%)', 'LYMPH%(%)', 'MONO%(%)', 'EO%(%)', 'BASO%(%)',
                    'IG#(10^3/uL)', 'IG%(%)', 'RET%(%)', 'RET#(10^9/L)', 'IRF(%)', 'LFR(%)', 'MFR(%)',
                    'HFR(%)', 'RET-He(pg)', '[PLT-I(10^3/uL)]', '[MicroR(%)]', '[MacroR(%)]',
                    '[TNC(10^3/uL)]', '[WBC-N(10^3/uL)]', '[TNC-N(10^3/uL)]', '[BA-N#(10^3/uL)]',
                    '[BA-N%(%)]', '[WBC-D(10^3/uL)]', '[TNC-D(10^3/uL)]', '[NEUT#&(10^3/uL)]',
                    '[NEUT%&(%)]', '[LYMP#&(10^3/uL)]', '[LYMP%&(%)]', '[HFLC#(10^3/uL)]',
                    '[HFLC%(%)]', '[BA-D#(10^3/uL)]', '[BA-D%(%)]', '[NE-SSC(ch)]',
                    '[NE-SFL(ch)]', '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]',
                    '[LY-Z(ch)]', '[MO-X(ch)]', '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]',
                    '[NE-WZ]', '[LY-WX]', '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]',
                    '[RBC-O(10^6/uL)]', '[PLT-O(10^3/uL)]', '[RBC-He(pg)]', '[Delta-He(pg)]',
                    '[RET-Y(ch)]', '[RET-RBC-Y(ch)]', '[IRF-Y(ch)]', '[FRC#(10^6/uL)]']

    unknown_values = ['ERROR', '----', '++++']
    cbc[numeric_data] = cbc[numeric_data].replace(unknown_values, np.NaN)
    cbc[numeric_data] = cbc[numeric_data].fillna(cbc.mean().iloc[0])

    # if Scaling is done on test set keep this section else remove
    numeric_df = pd.DataFrame(cbc[numeric_data])
    scaler = StandardScaler()
    standard_cbc = scaler.fit_transform(numeric_df)
    standard_cbc = pd.DataFrame(standard_cbc, columns=numeric_data)

    cbc[numeric_data] = standard_cbc
    # --------

    # keep1
    col_names = pd.Series(cbc.columns)
    # keep3
    omit_col_contining = ['WBC', 'LY', 'MO', 'NE', 'BA', 'EO', 'TNC', 'IG', 'HFLC', 'Left Shift', 'Ly']
    # keep2
    filter_col = col_names[col_names.str.contains('|'.join(omit_col_contining))]
    # keep
    keep_col = list(filter_col)
    cbc = cbc.drop(keep_col, axis=1)

    return cbc


def max_five(df):
    df = pd.melt(df)
    df['variable'] = df['variable'].map({0: 'HM', 1: 'SEP', 2: 'MDA', 3: 'AA',
                                         4: 'ITP', 5: 'NM', 6: 'MF+IDA', 7: 'IDA',
                                         8: 'MPY', 9: 'MA', 10: 'ITP+IDA', 11: 'CML-CP+IDA',
                                         12: 'CGD', 13: 'HCV', 14: 'ETS', 15: 'Hypersplenism',
                                         16: 'PV', 17: 'PRCA', 18: 'ACD', 19: 'PCA', 20: 'CDA',
                                         21: 'GA', 22: 'Extremly Increased Iron deposition'})
    df.rename(columns={'variable': 'Type'}, inplace=True)
    result = pd.DataFrame()
    while len(result) <= 5:
        df2 = pd.DataFrame()
        df2 = df[df['value'] == df['value'].max()]
        result = result.append(df2)
        idx = df.index.values[df['value'] == df['value'].max()]
        df = df.drop(idx)
        df = df.reset_index(drop=True)
    sum_other = df['value'].sum()
    result = result.append({"Type": "Others", 'value': sum_other}, ignore_index=True)
    return result


# creating app
app = Flask(__name__)
app.config['SECRET_KEY'] = "csrd-ned-nibd"

# Reading from json file
with open('config.json', 'r') as c:
    params = json.load(c)["params"]

# Upload folder
UPLOAD_FOLDER = 'E:/7th Semester/FYP/Projects/Github Repo/Flask-practice'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Creating by default endpoint
@app.route("/")
def default():
    # users = Users.query.filter_by().all() #No need to fetch user from database according to the requirement
    return render_template('index.html', params=params)


# Creating endpoint for home page
@app.route("/home")
def home():
    return render_template('home.html', params=params)


# CODE WITH BAD REQUEST ERROR FROM LOGIN TO PREDICTION PANEL
# Creating endpoint for prediction_panel that will be accessed following the login template
@app.route("/prediction_panel", methods=['GET', 'POST'])
def prediction_panel():
    global data
    if ('user' in session and session['user'] == params['admin_user']):  # Checking if the user is already logged in
        if request.method == 'POST':  # If user submit csv file
            f = request.files['chooseFile']  # get the file and store in variable
            f_name = f.filename
            split_extension = f_name.split('.')
            f_extension = split_extension[1]
            excel_ext = ['xls', 'xlsx', 'xlm']
            if f_extension in excel_ext:
                data = pd.read_excel(f)
            elif f_extension == 'csv':
                data = pd.read_csv(f)
            headers = data.columns  # obtain headers from csv file
            values = list(data.values)  # obtain rows from csv file and convert to list
            heading = "Obtained Dataset"  # Display dynamic header on the template
            btn_text = "View Results"
            return render_template('detection_panel.html', params=params, values=values, headers=headers,
                                   heading=heading, btn_text=btn_text)
        return render_template('detection_panel.html',
                               params=params)  # if user is already logged in, simply display detection panel

    if request.method == 'POST':  # if the user is not logged in then first ask for credential
        username = request.form.get('uname')  # Redirect to dashboard pannel
        userpass = request.form.get('user_pass')
        if (username == params['admin_user']) and (
                userpass == params['admin_password']):  # Checking whether the credentials are valid
            session[
                'user'] = username  # if credential are valid then Set the session variable, telling the flask app that this user is logged in
            if request.method == 'POST':  # If user submit csv file
                f = request.files['chooseFile']  # get the file and store in variable
                f_name = f.filename
                split_extension = f_name.split('.')
                f_extension = split_extension[1]
                excel_ext = ['xls', 'xlsx', 'xlm']
                if f_extension in excel_ext:
                    data = pd.read_excel(f)
                elif f_extension == 'csv':
                    data = pd.read_csv(f)
                headers = data.columns  # obtain headers from csv file
                values = list(data.values)  # obtain rows from csv file and convert to list
                heading = "Obtained Dataset"  # Display dynamic header on the template
                btn_text = "View Results"
                return render_template('detection_panel.html', params=params, values=values, headers=headers,
                                       heading=heading, btn_text=btn_text)

            return render_template('detection_panel.html',
                                   params=params)  # if user is successfully logged in, simply display detection panel

    return render_template('login.html', params=params)  # else render the login file


# # CODE THAT REMOVED BAD REQUEST ERROR FROM LOGIN TO PREDICTION PANEL
# @app.route("/prediction_panel", methods=['GET', 'POST'])
# def prediction_panel():
#     if request.method == 'POST':
#         if ('user' in session and session['user'] == params['admin_user']):     # Checking if the user is already logged in
#            # If user submit csv file
#             f = request.files['chooseFile'] # get the file and store in variable
#             data = pd.read_csv(f)           # read the file as csv
#             headers = data.columns          # obtain headers from csv file
#             values = list(data.values)      # obtain rows from csv file and convert to list
#             heading = "Obtained Dataset"    # Display dynamic header on the template
#             btn_text = "View Results"
#             return render_template('detection_panel.html', params=params, values=values, headers=headers, heading=heading,btn_text=btn_text)
#
#     elif ('user' not in session and session['user'] != params['admin_user']):
#         if request.method == 'POST':     # if the user is not logged in then first ask for credential
#             username = request.form.get('uname') # Redirect to dashboard pannel
#             userpass = request.form.get('user_pass')
#             if (username == params['admin_user']) and (userpass == params['admin_password']):         # Checking whether the credentials are valid
#                 session['user'] = username      # if credential are valid then Set the session variable, telling the flask app that this user is logged in
#
#
#                 return render_template('detection_panel.html', params=params) # if user is successfully logged in, simply display detection panel
#
#
#     return render_template('login.html', params=params)     # else render the login file


# Endpoint when user logout from prediction_panel
@app.route('/logout')
def logout():
    session.pop('user')  # Kill the user session
    # redirect to the main page
    return redirect('/')


# Endpoint for separate window of technical results.
@app.route('/tech_result')
def tech_result():
    return render_template('tech_results.html', params=params)


# route to html page - "table"
@app.route('/visual')
def visual():
    return render_template('visual.html', params=params)


# route to html page - "result"
@app.route('/result')
def result():
    global data
    if data.shape == (1, 139):
        new_data = processing_df(data)
        # note that "data" is a global variable and after the implementation of function
        #    processing_df(data) it has not changed rather the filtered dataframe is stored in
        #    a local variable named "new_data"

        values = list(new_data.values)
        prediction_df = pd.DataFrame(model.predict_proba(values))
        print(prediction_df)
        # selecting top 5 diseases with most probability
        visualiztion_results = max_five(prediction_df)
        print(visualiztion_results)

        # Visualization
        # bar chart
        fig1 = px.bar(visualiztion_results, x='Type', y='value', title='Types of Anemia with high value (Bar Chart)',
                      height=400, width=700)
        # fig.update_layout( paper_bgcolor="LightSteelBlue")
        fig1.update_traces(marker_color='rgb(188,128,189)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5,
                           opacity=0.6)
        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        # pie chart
        fig2 = px.pie(visualiztion_results, names='Type', values='value',
                      title='Types of Anemia with high value (Pie Chart)',
                      color_discrete_sequence=px.colors.sequential.Teal, height=400, width=700)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('result.html', params=params, visualiztion_results=visualiztion_results,
                               graphJSON=graphJSON, graphJSON2=graphJSON2)

    else:
        print('df not right')

    return render_template('result.html', params=params)


app.run()
