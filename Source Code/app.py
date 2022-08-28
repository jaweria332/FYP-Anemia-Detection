# Importing libraries
from flask import Flask, render_template, request, redirect, send_file, session
import pandas as pd
import numpy as np
import json
import pickle
import sys
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
import joblib

# creating app
app = Flask(__name__)
app.config['SECRET_KEY'] = "csrd-ned-nibd"



print('Python %s on %s' % (sys.version, sys.platform))


model = joblib.load('rf_model.pkl')

data = pd.DataFrame()


def processing_df(cbc):
    cbc = cbc.reset_index(drop=True)

    # removes all un-named columns from dataset
    cbc = cbc.loc[:, ~cbc.columns.str.contains('^Unnamed')]

    # removing patient personal info columns
    cbc = cbc.drop(['Nickname', 'Analyzer ID', 'Date', 'Time', 'Rack', 'Position',
                    'Sample No.', 'Sample Inf.', 'Order Type', 'Reception Date',
                    'Measurement Mode', 'Patient ID', 'Analysis Info.', 'Error(Func.)',
                    'Error(Result)', 'Order Info.', 'WBC Info.', 'PLT Info.',
                    'Rule Result', 'Validate', 'Validator',
                    'Action Message (Check)', 'Action Message (Review)',
                    'Action Message (Retest)', 'Sample Comment', 'Patient Name',
                    'Birth', 'Sex', 'Patient Comment', 'Ward Name', 'Doctor Name',
                    'Output', 'Sequence No.', 'Discrete', 'Q-Flag(Blasts/Abn Lympho?)',
                    'Q-Flag(Blasts?)', 'Q-Flag(Abn Lympho?)'], axis=1)

    # grouping numeric categorical columns
    catag_data = ['WBC Abnormal', 'WBC Suspect', 'RBC Abnormal', 'RBC Suspect',
                  'PLT Abnormal', 'PLT Suspect', 'IP ABN(WBC)WBC Abn Scattergram',
                  'IP ABN(WBC)Neutropenia',
                  'IP ABN(WBC)Neutrophilia', 'IP ABN(WBC)Lymphopenia',
                  'IP ABN(WBC)Lymphocytosis', 'IP ABN(WBC)Monocytosis',
                  'IP ABN(WBC)Eosinophilia', 'IP ABN(WBC)Basophilia',
                  'IP ABN(WBC)Leukocytopenia', 'IP ABN(WBC)Leukocytosis',
                  'IP ABN(WBC)NRBC Present', 'IP ABN(WBC)IG Present',
                  'IP ABN(RBC)RBC Abn Distribution', 'IP ABN(RBC)Dimorphic Population',
                  'IP ABN(RBC)Anisocytosis', 'IP ABN(RBC)Microcytosis',
                  'IP ABN(RBC)Macrocytosis', 'IP ABN(RBC)Hypochromia',
                  'IP ABN(RBC)Anemia', 'IP ABN(RBC)Erythrocytosis',
                  'IP ABN(RBC)RET Abn Scattergram', 'IP ABN(RBC)Reticulocytosis',
                  'IP ABN(PLT)PLT Abn Distribution', 'IP ABN(PLT)Thrombocytopenia',
                  'IP ABN(PLT)Thrombocytosis', 'IP ABN(PLT)PLT Abn Scattergram',
                  'IP SUS(WBC)Blasts/Abn Lympho?', 'IP SUS(WBC)Blasts?',
                  'IP SUS(WBC)Abn Lympho?', 'IP SUS(WBC)Left Shift?',
                  'IP SUS(WBC)Atypical Lympho?', 'IP SUS(RBC)RBC Agglutination?',
                  'IP SUS(RBC)Turbidity/HGB Interf?', 'IP SUS(RBC)Iron Deficiency?',
                  'IP SUS(RBC)HGB Defect?', 'IP SUS(RBC)Fragments?',
                  'IP SUS(PLT)PLT Clumps?']

    cbc[catag_data] = cbc[catag_data].fillna(value=0)

    # grouping descriptive categorical columns
    cbc['Judgment'] = cbc['Judgment'].map({'Positive': 1, 'Negative': 0})

    cbc['Positive(Diff.)'] = cbc['Positive(Diff.)'].map({'Diff.': 1})

    cbc['Positive(Morph.)'] = cbc['Positive(Morph.)'].map({'Morph.': 1})

    cbc['Positive(Count)'] = cbc['Positive(Count)'].map({'Count': 1})

    # replacing all NaN values by 0 as per suggested by the domain expert
    cbc[['Judgment', 'Positive(Diff.)', 'Positive(Count)', 'Positive(Morph.)']] = cbc[
        ['Judgment', 'Positive(Diff.)', 'Positive(Count)', 'Positive(Morph.)']].fillna(value=0)

    # grouping columns having signed values only
    signed_col = list(cbc.loc[:, cbc.columns.str.contains('/M')])
    cbc[signed_col] = cbc[signed_col].replace('+', 1)
    cbc[signed_col] = cbc[signed_col].replace('-', -1)
    cbc[signed_col] = cbc[signed_col].replace(np.NaN, 0)

    # grouping (continuous) numeric data columns
    numeric_data = ['Q-Flag(Left Shift?)', 'Q-Flag(Atypical Lympho?)', 'Q-Flag(RBC Agglutination?)',
                    'Q-Flag(Turbidity/HGB Interf?)', 'Q-Flag(Iron Deficiency?)', 'Q-Flag(HGB Defect?)',
                    'Q-Flag(Fragments?)', 'Q-Flag(PLT Clumps?)', 'WBC(10^9/L)', 'RBC(10^12/L)',
                    'HGB(g/dL)', 'HCT(%)', 'MCV(fL)', 'MCH(pg)', 'MCHC(g/dL)', 'PLT(10^3/uL)',
                    'RDW-SD(fL)', 'RDW-CV(%)', 'PDW(fL)', 'MPV(fL)', 'P-LCR(%)', 'PCT(%)',
                    'NRBC#(10^3/uL)', 'NRBC%(%)', 'NEUT#(10^3/uL)', 'LYMPH#(10^3/uL)', 'MONO#(10^3/uL)',
                    'EO#(10^3/uL)', 'BASO#(10^3/uL)', 'NEUT%(%)', 'LYMPH%(%)', 'MONO%(%)', 'EO%(%)',
                    'BASO%(%)', 'IG#(10^3/uL)', 'IG%(%)', 'RET%(%)', 'RET#(10^9/L)', 'IRF(%)',
                    'LFR(%)', 'MFR(%)', 'HFR(%)', 'RET-He(pg)', 'IPF(%)', '[PLT-I(10^3/uL)]',
                    '[MicroR(%)]', '[MacroR(%)]', '[TNC(10^9/L)]', '[WBC-N(10^9/L)]', '[TNC-N(10^9/L)]',
                    '[BA-N#(10^3/uL)]', '[BA-N%(%)]', '[WBC-D(10^9/L)]', '[TNC-D(10^9/L)]',
                    '[NEUT#&(10^3/uL)]', '[NEUT%&(%)]', '[LYMP#&(10^3/uL)]', '[LYMP%&(%)]',
                    '[HFLC#(10^3/uL)]', '[HFLC%(%)]', '[BA-D#(10^3/uL)]', '[BA-D%(%)]', '[NE-SSC(ch)]',
                    '[NE-SFL(ch)]', '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]',
                    '[MO-X(ch)]', '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]',
                    '[LY-WX]', '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]', '[WBC-P(10^9/L)]',
                    '[TNC-P(10^9/L)]', '[RBC-O(10^12/L)]', '[PLT-O(10^3/uL)]', '[RBC-He(pg)]',
                    '[Delta-He(pg)]', '[RET-Y(ch)]', '[RET-RBC-Y(ch)]', '[IRF-Y(ch)]', '[FRC#(10^12/L)]',
                    '[FRC%(%)]', '[HYPO-He(%)]', '[HYPER-He(%)]', '[RPI]', '[RET-UPP]', '[RET-TNC]',
                    '[PLT-F(10^3/uL)]', '[H-IPF(%)]', '[IPF#(10^3/uL)]', 'WBC-BF(10^3/uL)',
                    'RBC-BF(10^6/uL)', 'MN#(10^3/uL)', 'PMN#(10^3/uL)', 'MN%(%)', 'PMN%(%)',
                    'TC-BF#(10^3/uL)', '[HF-BF#(10^3/uL)]', '[HF-BF%(/100WBC)]', '[NE-BF#(10^3/uL)]',
                    '[NE-BF%(%)]', '[LY-BF#(10^3/uL)]', '[LY-BF%(%)]', '[MO-BF#(10^3/uL)]', '[MO-BF%(%)]',
                    '[EO-BF#(10^3/uL)]', '[EO-BF%(%)]', '[RBC-BF2(10^6/uL)]', 'HPC#(10^3/uL)',
                    '[HGB-O(g/dL)]', '[PLT-F2(10^3/uL)]', 'IP SUS(RBC)pRBC?', 'Q-Flag(pRBC?)',
                    '[Delta-HGB(g/dL)]', '[MCHC-O(g/dL)]', '[WBC(10^3/uL)]', '[RBC(10^6/uL)]',
                    '[RBC-I(10^6/uL)]', '[RBC-O(10^6/uL)]', '[NEUT#(10^3/uL)]', '[LYMPH#(10^3/uL)]',
                    '[MONO#(10^3/uL)]', '[EO#(10^3/uL)]', '[NEUT%(%)]', '[LYMPH%(%)]', '[MONO%(%)]',
                    '[EO%(%)]', '[MN#(10^3/uL)]', '[PMN#(10^3/uL)]', '[HF#(10^3/uL)]', '[MN%(%)]',
                    '[PMN%(%)]', '[HF%(/100WBC)]', '[TC#(10^3/uL)]', '[HPC%(%)]']
    unknown_values = ['ERROR', '----', '++++', '*', '@', '    ']
    cbc[numeric_data] = cbc[numeric_data].replace(unknown_values, np.NaN)

    # for filling null values in column that have some numeric values in it i.e. some mean can be generated
    cbc[numeric_data] = cbc[numeric_data].fillna(value=cbc[numeric_data].mean())

    # for filling null values in column that have no numeric values in it i.e. no mean can be generated so replacing it by 0
    cbc[numeric_data] = cbc[numeric_data].fillna(value=0)

    # scaling all parameters in order to have equal weightage in the model training
    numeric_df = pd.DataFrame(cbc[numeric_data])
    scaler = StandardScaler()
    standard_cbc = scaler.fit_transform(numeric_df)
    standard_cbc = pd.DataFrame(standard_cbc, columns=numeric_data)

    cbc[numeric_data] = standard_cbc

    # omitting un-needed column from dataset
    col_names = pd.Series(cbc.columns)
    omit_col_contining = ['TNC', '/M']
    filter_col = col_names[col_names.str.contains('|'.join(omit_col_contining))]
    keep_col = list(filter_col)
    cbc = cbc.drop(keep_col, axis=1)
    return cbc



def max_five(df):

    df = pd.melt(df)

    df['variable'] = df['variable'].map({0: 'Anemia' , 1: 'IDA' , 2: 'MA' ,
        3: 'SCA' , 4: 'AA' , 5: 'SA' , 6: 'MDA' , 7: 'PTA' ,
        8: 'T_Min' , 9: 'HA' , 10: 'ITP' , 11: 'T_Maj' , 12: 'Infection' ,
        13: 'PRCA' , 14: 'BMF' , 15: 'AML' , 16: 'CML' ,
        17: 'MS' , 18: 'CLL' , 19: 'ALL' , 20: 'PC' ,
        21: 'ET' , 22: 'MCL' , 23: 'Malaria' , 24: 'HL' ,
        25: 'GT' , 26: 'HCL' , 27: 'MM' , 29: 'PM' , 38: 'MZL' ,
        91: 'WAHA' , 93: 'TTP' , 94: 'PNH' , 99: 'HC' ,
        101: 'BP' , 102: 'NL' , 103: 'MT' , 104: 'PP' ,
        105: 'RTWA' , 106: 'AWLP' , 107: 'NN',
        108: 'ST' , 109: 'TI' , 110: 'API' ,
        111: 'PCD' , 112: 'AD' , 113: 'Sev_Anemia' ,
        115: 'Dengue' , 116: 'HS' , 153: 'AMLM3' ,
        154: 'AMLM4' , 161: 'CMLCP' , 163: 'CMLBP' ,
        164: 'TCTA' , 165: 'TML' , 999: 'KCDNF' })


    df.rename(columns={'variable': 'Type'}, inplace = True)

    result = pd.DataFrame()
    while len(result) < 5:
        df2 = pd.DataFrame()
        df2 = df[df['value'] == df['value'].max()]
        result = result.append(df2)
        idx = df.index.values[df['value'] == df['value'].max()]
        df = df.drop(idx)
        df = df.reset_index(drop=True)

    # ========================================================================================
    # Changing to remove the others bar chart value
    # sum_other = df['value'].sum()
    # result = result.append({"Type": "Others", 'value': sum_other}, ignore_index=True)
    # result = result.append({"Type": "Others"}, ignore_index=True)
    return result






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




# Creating endpoint for prediction_panel that will be accessed
@app.route("/prediction_panel", methods=['GET', 'POST'])
def prediction_panel():
    global data
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
    return render_template('detection_panel.html', params=params)  # if user is already logged in, simply display detection panel





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
    if data.shape == (1, 433):
        new_data = processing_df(data)
        # note that "data" is a global variable and after the implementation of function
        #    processing_df(data) it has not changed rather the filtered dataframe is stored in
        #    a local variable named "new_data"

        values = list(new_data.values)
        prediction_df = pd.DataFrame(model.predict_proba(values), columns=model.classes_)
        print(prediction_df)
        # selecting top 5 diseases with most probability
        visualiztion_results = max_five(prediction_df)
        print(visualiztion_results)

        # Visualization
        # column chart
        fig1 = px.bar(visualiztion_results, x='Type', y='value', color="Type",
                      text = visualiztion_results['value'], color_discrete_sequence=px.colors.qualitative.Pastel, height=400, width=700)
        fig1.update_layout(yaxis={'categoryorder':'total descending'},font=dict(size=15))
        # fig.update_layout( paper_bgcolor="LightSteelBlue")
        #fig1.update_traces(marker_color='rgb(188,128,189)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        # pie chart
        fig2 = px.pie(visualiztion_results, names='Type', values='value',
                     
                      color="Type", color_discrete_sequence=px.colors.qualitative.Pastel, height=400, width=700)
        fig2.update_layout(font=dict(size=15))
        fig2.update_traces(textposition='inside', textinfo='label')
        graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        # bar chart
        fig3 = px.bar(visualiztion_results, x='value', y='Type',
                      text = visualiztion_results['value'], color_discrete_sequence=px.colors.qualitative.Pastel, height=450, width=700, orientation='h', color="Type")
        fig3.update_layout(yaxis={'categoryorder':'total descending'},font=dict(size=15))
        #fig3.update_traces(marker_color='rgb(188,128,189)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5,opacity=0.6)
        #fig3.update_traces(textposition='inside', textinfo='percent+label')
        graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        #donut chart
        fig4 = px.pie(visualiztion_results, values="value", names="Type", hole=.3, color="Type", color_discrete_sequence=px.colors.qualitative.Pastel,height=400, width=700)
        fig4.update_traces(textposition='inside', textinfo='label')
        fig4.update_layout(font=dict(size=15))
        #fig4 = px.line(visualiztion_results, x="value", y="Type", markers=True, text = visualiztion_results['value'])
        graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('result.html', params=params, visualiztion_results=visualiztion_results,
                               graphJSON=graphJSON, graphJSON2=graphJSON2, graphJSON3=graphJSON3, graphJSON4=graphJSON4)
    else:
        error_statement = "Entered file does not have correct amount of parameters!"
        return render_template('error.html', error_statement=error_statement)

    return render_template('result.html', params=params)


app.run()
