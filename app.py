# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:01:10 2020

@author: Jiwoo Ahn
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output , State

import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import mplstereonet
from mplstereonet import plane_intersection, plunge_bearing2pole, rake

# Initiate the app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'AusCovidDash'

colors = {
    'background': '#000000',
    'text': '#5d76a9',
    'label': '#f5b112'
}

def dipdir2strike(dd):
    strike = dd - 90
    if strike < 0:
        strike = strike + 360
    return strike

def strike2dipdir(st):
    dd = st + 270
    if dd > 359:
        dd = dd - 360
    return dd

def pole2plane(trend, plunge):
    dipdir = trend - 180
    if dipdir < 0:
        dipdir = dipdir + 360
    dip = 90 - plunge
    return dipdir, dip

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours
    
def stereoplot(Dip, DipDirection, FrictionAngle, figsize,):
    
    """ 
    
    Plot histograms with best fit probability density functions
    
    :param list Dip: list containing Dip angles
    
    :param list DipDirection: list containing Dip Directions
    
    :param tuple(float,float) figsize: figure size width,height
    
    """
    # Figure settings
    plt.rcParams.update({'font.size': 12})
    matplotlib.style.use('seaborn-whitegrid')
    fig = plt.figure (figsize=figsize, dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='stereonet')
    
    # Convert dip to strike
    strikes = [dipdir2strike(x) for x in DipDirection]
    
    colours = ['r','g','b']
    i = 0
    # Set what the pole markers look like
    for x,y in zip(strikes,Dip):
        ax.pole(x, y, color=colours[i], marker='D', markersize=10, alpha=1, zorder=5)
        ax.plane(x, y, color=colours[i], linewidth=2)
        ax.rake(x, y, 90, color=colours[i], marker='o',markersize=10, alpha=1, zorder=5)
        i = i + 1
    
    # Stereonet overlay settings
    ax.set_azimuth_ticks([0,90,180,270],labels=['N','E','S','W'])
    
    # angle = np.linspace(0,360,36,False)
    # for i in angle:
    #     ax.pole([i,i],[10,90],'silver','-', zorder=1)
    #     angle2 = np.linspace(10,90,8,False)
    #     for i in angle2:
    #         ax.cone(plunge=90, bearing=20, angle=i, facecolors='none', edgecolors='silver', bidirectional=False, zorder=1)

    # Plane intersections   
    ints = plane_intersection([strikes[0],strikes[0],strikes[1]],[Dip[0],Dip[0],Dip[1]],[strikes[1],strikes[2],strikes[2]],[Dip[1],Dip[2],Dip[2]])
    ints_conv = plunge_bearing2pole(ints[0],ints[1])
    ax.pole(ints_conv[0],ints_conv[1], color='k', marker='o', markersize='10')
    
    # Friction circle
    ax.cone(90,0,90-FrictionAngle, alpha=0.1, color='r', zorder=1, bidirectional=False)
    
    ax.grid(True, kind='polar')
    plt.tight_layout()
    
    Dips = [Dip, 90-ints_conv[1]]
    Joints = ['1','2','3','1/2','1/3','2/3']
    
    if np.max(Dips) >= FrictionAngle:
        Stability = 'Unstable'
        Mode = "Sliding on Joint {0}".format(Joints[np.argmax(Dips)])
    else:
        Stability = 'Stable'
        Mode = 'N/A'
    
    # Table in top plot
    table_data=[
        ["Friction Angle", "{0}°".format(FrictionAngle)],
        ["Joints", "Dip / Dip Direction"],
        ["1", "{0}°/{1:0=3d}°".format(int(round(Dip[0])),int(round(DipDirection[0])))],
        ["2", "{0}°/{1:0=3d}°".format(int(round(Dip[1])),int(round(DipDirection[1])))],
        ["3", "{0}°/{1:0=3d}°".format(int(round(Dip[2])),int(round(DipDirection[2])))],
        ["1/2", "{0}°/{1:0=3d}°".format(int(90-round(ints_conv[1][0])), int(round(strike2dipdir(ints_conv[0][0]))))],
        ["1/3", "{0}°/{1:0=3d}°".format(int(90-round(ints_conv[1][1])), int(round(strike2dipdir(ints_conv[0][1]))))],
        ["2/3", "{0}°/{1:0=3d}°".format(int(90-round(ints_conv[1][2])), int(round(strike2dipdir(ints_conv[0][2]))))],
        ["Stability", Stability],
        ["Mode", Mode]
        ]
    
    table = ax.table(cellText=table_data, bbox = [1.1, 0.3, 0.4, 0.35], cellLoc='center', loc='right', colWidths=[0.2,0.2])# , loc='top right')#, colWidths = [0.4]*2)
    for (row, col), cell in table.get_celld().items():
        if (row == 0 or row ==1 or row == 8 or row == 9):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        if (row == 2):
            cell.set_text_props(color='r')
        if (row == 3):
            cell.set_text_props(color='g')
        if (row == 4):
            cell.set_text_props(color='b')
    
    return fig, ax


# Server
app.layout = html.Div([
    html.H1(children='Kinematic Analysis of Underground Rock Wedge',
            style={'textAlign': 'center','font-family':'Verdana','color': colors['text'],'padding-top': 20}),
    html.P(children='''Joint Orientations (Dip / Dip Direction)''',
           style={'textAlign': 'center','font-size':24,'font-family':'Verdana','color': colors['text'],'padding-bottom': 10}),
    html.Div([html.Label(["Joint 1",dcc.Input(id='joint1_dip-state', type='number', min=1, max=89, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'}),dcc.Input(id='joint1_dd-state', type='number', min=0, max=359, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Label(["Joint 2",dcc.Input(id='joint2_dip-state', type='number', min=1, max=89, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'}),dcc.Input(id='joint2_dd-state', type='number', min=0, max=359, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Label(["Joint 3",dcc.Input(id='joint3_dip-state', type='number', min=1, max=89, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'}),dcc.Input(id='joint3_dd-state', type='number', min=0, max=359, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Label(["Friction Angle (°)",dcc.Input(id='fric_ang-state', type='number', min=1, max=89, value=30, style={'width': '50px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Button('Update Stereonet', id='update_button-state', n_clicks=0)],
             style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div(html.Img(id='graph',style={'width':'60%'}),style={'vertical-align':'middle', 'textAlign':'center'}),
    html.Div(dcc.Markdown('''
       
    _Created by : Jiwoo Ahn_
    
    [Github Repo](https://github.com/j-ahn/RockWedge)
    
    '''), style = {'font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
])

@app.callback(
    Output('graph', 'src'),
    [Input('update_button-state', 'n_clicks')],
    [State('joint1_dip-state', 'value')],
    [State('joint2_dip-state', 'value')],
    [State('joint3_dip-state', 'value')],
    [State('joint1_dd-state', 'value')],
    [State('joint2_dd-state', 'value')],
    [State('joint3_dd-state', 'value')],
    [State('fric_ang-state', 'value')]
)

#   joint1_dip, joint1_dd, joint2_dip, joint2_dd, joint3_dip, joint3_dd, fric_ang
def update_figure(n_clicks, joint1_dip, joint2_dip, joint3_dip, joint1_dd, joint2_dd, joint3_dd, fric_ang):
    if n_clicks > 0 and joint1_dip and joint1_dd and joint2_dip and joint2_dd and joint3_dip and joint3_dd and fric_ang:
        Dip = [joint1_dip, joint2_dip, joint3_dip]
        DipDirection = [joint1_dd, joint2_dd, joint3_dd]
        print(Dip)
        print(DipDirection)
        FrictionAngle = fric_ang
        
        # Stereonet on data
        stereofig, stereoax = stereoplot(Dip, DipDirection, FrictionAngle, (10,10))
        buf = io.BytesIO() # in-memory files
        stereofig.savefig(buf, format = "png", dpi=200, bbox_inches = "tight") # save to the above file object
        data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        plt.close()
        return "data:image/png;base64,{}".format(data)

if __name__ == "__main__":
    app.run_server()
    
                          