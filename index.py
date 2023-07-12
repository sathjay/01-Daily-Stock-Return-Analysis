import pandas as pd     #(version 1.0.0)
import plotly    #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px
import plotly.figure_factory as ff
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output,State
from dash import dash_table
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime
import scipy as sc
import plotly.offline as pyo
import plotly.graph_objs as go
import os
from plotly.subplots import make_subplots
from scipy import stats
import scipy.stats as stats

SP500 = pd.read_csv('data/Company_Name_List.csv')
SP500.set_index('Symbol', inplace=True)  
# The reason why Symbol is set as index is that in the next step, fetching the name of company is easy.
stock_dropdown_list=[]
for symbol in SP500.index:
    stock_dropdown_list.append({'label':'{} - {}'.format(symbol,SP500.loc[symbol]['Name']), 'value':symbol})
    
start_date=dt(2015, 1, 1).date()  # Default start date to get the price quotes
end_date = datetime.datetime.now().date()
dt = pd.to_datetime(start_date, format='%Y-%m-%d')
dt1 = pd.to_datetime(end_date, format='%Y-%m-%d')
num_of_days_btw_start_and_end= (dt1-dt).days
number_of_years = num_of_days_btw_start_and_end/365

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None,plot_bgcolor='rgb(173,216,230)' , paper_bgcolor= 'rgb(173,216,230)' )
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

meta_tags = [{'name': 'viewport', 'content': 'width=device-width'}]
external_stylesheets = [ meta_tags]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,title = "Daily Return Analysis")
server = app.server

app.layout = html.Div([
    
            html.H4('Stock Daily Return Analysis:'),
            
            html.P('For stock selected from the below drop down, users will see stock return summary, price chart, \
                    Daily percentage price change over the duration (Volatility Clusters), Histogram of Daily Percent Return, \
                    QQ Plot and KS statistics test to check whether the Daily percentage price change are Normal Gaussian Distribution'),
            
            html.Label(['Select Company/EFT from dorp down:'],
                   style={'font-weight': 'bold'}),
        
            html.Br(),
            
            dcc.Dropdown(id='my_selected_stock',
                    options=stock_dropdown_list,
                    optionHeight=35,
                    value = 'SPY',
                    disabled=False,                     #disable dropdown value selection
                    multi=False,                        #allow multiple dropdown values to be selected
                    searchable=True,                    #allow user-searching of dropdown values
                    placeholder='Please select...',     #gray, default text shown when no option is selected
                    clearable=True,                     #allow user to removes the selected value
                    className='dropdown_box',           #activate separate CSS document in assets folder
                    ),   
            
            html.Hr(),
            html.H4(id='new_string'),
            html.Br(),
            html.Div(
            dash_table.DataTable(id = 'summary_details',
                                 style_cell={'textAlign': 'center'},
                                 style_header={'backgroundColor': 'blue',
                                                'fontWeight': 'bold',
                                                'color':'white',
                                                'fontSize': '16px',
                                                'whiteSpace': 'normal',
                                                'height': 'auto',
                                                },
                                 style_data={
                                            'color': 'black',
                                            'backgroundColor': 'white',
                                            'fontSize': '14px',
                                            }),className = 'table_container'),
            
            html.Br(),
            
            dcc.Graph(id='price_chart',figure = blank_fig(),
                      style={'width': '90%', 'height': '80vh'}),
            
            html.Hr(),   
            
            dcc.Graph(id='daily_return_over_time',figure = blank_fig(),
                     style={'width': '90%', 'height': '80vh'}),
            
            html.Hr(),   
            
            dcc.Graph(id='daily_return_distribution',figure = blank_fig(),
                     style={'width': '90%', 'height': '80vh'}),
            
            html.Hr(),  
            
            dcc.Graph(id='QQ_Plot',figure = blank_fig(),
                      style={'width': '90%', 'height': '80vh'}),
            
            html.H4(id='is_normal'),
        
            html.P(id='ks_stat'),
            
            html.Hr(),
           
            
    
            ],className = 'main_container')


@app.callback( [
                Output('new_string', 'children'),
                Output('summary_details',component_property = 'data'),
                Output('summary_details',component_property = 'columns'),
                Output('price_chart',component_property = 'figure'),
                Output('daily_return_over_time',component_property = 'figure'),
                Output('daily_return_distribution',component_property = 'figure'),
                Output('QQ_Plot',component_property = 'figure'),
                Output('is_normal', 'children'),
                Output('ks_stat', 'children'),
                
               ],
              [Input('my_selected_stock','value')])

def update_layout(my_selected_stock):
    
    df= yf.download(my_selected_stock,period = '1d',start = start_date, end = end_date)
    
    df_close = df[['Adj Close']]
    
    # Calculating Daily Return 
    simple_return = df_close.pct_change()
    simple_return = simple_return.dropna()
    simple_return.rename(columns = {'Adj Close':'Daily Return'}, inplace = True)
    simple_return['Daily Return in Percent'] = simple_return['Daily Return'].apply(lambda x: x*100)
    simple_return['Date'] = simple_return.index
    simple_return.drop(['Daily Return'], axis=1, inplace = True)
    
    
    # Calculation for summary table:
    
    current_close = df_close.iloc[-1]['Adj Close']
    previous_month_close = df_close.iloc[-22]['Adj Close']
    previous_3_month_close = df_close.iloc[-66]['Adj Close']
    previous_year_close = df_close.iloc[-252]['Adj Close']
    previous_3year_close =  df_close.iloc[-756]['Adj Close']
    
    one_month_return =   round((((current_close - previous_month_close)/previous_month_close)*100),2)
    three_month_return = round((((current_close - previous_3_month_close)/previous_3_month_close)*100),2)
    one_year_return = round((((current_close - previous_year_close )/previous_year_close)*100),2)
    three_year_return = round((((current_close - previous_3year_close )/previous_3year_close)*100),2)
    
    mean_daily_return = simple_return.mean()
    daily_return_std = simple_return.std()
    daily_return_std = daily_return_std['Daily Return in Percent'] 
    annualized_volatility = daily_return_std*(252**.5)
    CAGR = (pow((df.iloc[-1]['Adj Close']/df.iloc[0]['Adj Close']),(1/number_of_years))-1)*100
    
    annualized_volatility =  round(annualized_volatility,2)
    daily_return_std = round(daily_return_std,3)
    CAGR = round(CAGR,2)
    
    summary_details = {
    
    'Stock' : my_selected_stock,
    'Current Close': current_close,
    '1 Month Return %': one_month_return,
    '3 Month Return %': three_month_return,
    '1 Year Return %': one_year_return,
    '3 Year Return %': three_year_return,
    'CAGR %': CAGR,
    'Annualized Volatility %': annualized_volatility
        
                        }
    
    summary_df = pd.DataFrame(summary_details,index=[0])
    
    print(summary_df)
    a= "{} Summary:".format(my_selected_stock)
    s_columns = [{'name':col, 'id':col} for col in summary_df.columns]
    s_data = summary_df.to_dict(orient = 'records')
    
    
    # Price Chart
    data =[]
    name = my_selected_stock
    trace = go.Scatter(x=df.index.values,
                       y=df['Close'],
                       mode='lines',
                       name= name
                       )
    data.append(trace)
    layout = go.Layout(title= '{} Price Chart'.format(name),
                       plot_bgcolor='white',
                       xaxis=dict(showline = True,linecolor='black',showgrid = False),
                       yaxis=dict(showline = True,linecolor='black',showgrid = True,gridwidth=1,gridcolor='black'),
                       )
    fig1 = go.Figure(data=data,layout=layout)
    
    a= "{} Return Summary:".format(my_selected_stock)
    
    # Daily Return Scatter chart over time
    
    fig2 = px.scatter(simple_return, 
                 x="Date", 
                 y="Daily Return in Percent", 
                 title = '{} Daily Percent Return (Volatility Clusters)'.format(name),
                 color="Daily Return in Percent",
                 color_continuous_scale=["red", "yellow", "green"]
                 )
    
    # Daily Return Distribution
    
    fig3 = px.histogram(simple_return, 
                        x="Daily Return in Percent",
                       marginal="box",
                       hover_data=simple_return.columns,
                       title = '{} Daily Percent Return Distribution'.format(name)
                       )
       
    # Q-Q or Quantile-Quantile Plots
    # It plots two sets of quantiles against one another i.e. theoretical quantiles against the actual quantiles 
    # of the variable.
    
    daily_simple_return = simple_return['Daily Return in Percent']
    
    fig4 = go.Figure()
    
    qq = stats.probplot(daily_simple_return, dist='norm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    fig4.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
    fig4.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')

    fig4['layout'].update({
                        'title': 'Quantile-Quantile Plot',
                        'plot_bgcolor':'white',
                        'xaxis': {
                                'title': 'Theoritical Quantities',
                                'showline' : True,
                                'linecolor':'black',
                                'zeroline':False,
                                },
                        'yaxis': {
                                'title': 'Sample Quantities',
                                'showline' : True,
                                'zeroline':True,
                                'zerolinewidth':1, 
                                'zerolinecolor':'black',
                                'linecolor':'black',
                                'showgrid' : True,
                                'gridwidth':1,
                                'gridcolor':'black',
                                
                                },
                        'showlegend': False
    
                    })


    b= 'Are Daily Returns Normal Gaussian Distribution?'
    
    
    ks_statistic, p_value = stats.kstest(daily_simple_return, 'norm', args = (daily_simple_return.mean(), daily_simple_return.std()))
    print(ks_statistic, p_value)
    if p_value > 0.05:
        c = 'KS_statistic value is {} and p value is {}. Since p value is greater than 0.05 hence this distribution is Probably Gaussian'.format(ks_statistic,p_value)
    else:
        c = 'KS_statistic value is {} and p value is {}. Since p value is less than 0.05 hence this distribution is Probably Not Gaussian'.format(ks_statistic,p_value)
    
    
    
    
    
    
    return a,s_data,s_columns,fig1,fig2,fig3,fig4,b,c,




if __name__ == '__main__':
    app.run_server(debug=True)

