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
from scipy.spatial import Delaunay
from scipy.optimize import linprog
import mplstereonet
from mplstereonet.stereonet_math import plane_intersection, plunge_bearing2pole, pole, rake, plane, geographic2pole

# Initiate the app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'Rock Wedge'

colors = {
    'background': '#000000',
    'text': '#5d76a9',
    'label': '#f5b112'
}


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)

    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second

def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst

def norm_ang(ang):
    return (360 + ang % 360) % 360

def ang_between(n, a, b):
    n = norm_ang(n)
    a = (3600000 + a) % 360
    b = (3600000 + b) % 360
    
    if (a < b):
        if a <= n and n <= b:
            return True
        else:
            return False
    elif a<= n or n <= b:
        return True
    else:
        return False
        
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

def point_inside_polygon(x, y, poly, include_edges=True):
    '''
    Test if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as 
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times. 
    Works fine for non-convex polygons.
    '''
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                elif x < min(p1x, p2x):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    inside = include_edges
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside

def stereoplot(Dip, DipDirection, FrictionAngle, figsize,):
    
    """ 
    
    Plot histograms with best fit probability density functions
    
    :param list Dip: list containing Dip angles
    
    :param list DipDirection: list containing Dip Directions
    
    :param tuple(float,float) figsize: figure size width,height
    
    """
    # Figure settings
    plt.rcParams.update({'font.size': 12})
    #matplotlib.style.use('seaborn-whitegrid')
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
    
    # Plane intersections   
    plunge, bearing = plane_intersection([strikes[0],strikes[0],strikes[1]],[Dip[0],Dip[0],Dip[1]],[strikes[1],strikes[2],strikes[2]],[Dip[1],Dip[2],Dip[2]])
    plunge = [x for x in plunge]
    bearing = [x for x in bearing]
    ax.line(plunge, bearing, color='k', marker='o', markersize='10')
    
    # Convert plane intersections as poles (strike/dip)
    int_strike, int_dip = plunge_bearing2pole(plunge, bearing)
    int_strike = [x for x in int_strike]
    int_dip = [x - 0.001 for x in int_dip]
    joint_strikes = [x + 180 for x in strikes]
    joint_dips = [90- x for x in Dip]
    point_strikes = joint_strikes + int_strike
    point_dips = joint_dips + int_dip
    real_dips = Dip + plunge
    
    # Wedge Shaded area
    intersects = [[0,1],[0,2],[1,2]]
    j_inside_strikes = []
    j_inside_dips = []
    for joint, intersect in enumerate(intersects):
        j_plane = plane(strikes[joint],Dip[joint])
        j_plane = np.vstack((j_plane[0].T,j_plane[1].T))
        
        j_plane_poles = geographic2pole(j_plane[0], j_plane[1])
        
        int1 = int_strike[intersect[0]]
        int2 = int_strike[intersect[1]]
        int_diff = int1-int2
        
        if int_diff >= -180 and int_diff < 0 or int_diff > 180:
            ints_ordered = [int1, int2]
        elif int_diff <= 180 and int_diff > 0 or int_diff < -180:
            ints_ordered = [int2, int1]
        
        j_inside_strike = []
        j_inside_dip = []
        
        for i, x in enumerate(j_plane_poles[0]):
            if ang_between(x, ints_ordered[0], ints_ordered[1]):
                j_inside_strike.append(j_plane_poles[0][i])
                j_inside_dip.append(j_plane_poles[1][i])
        
        j_inside_strikes.append(j_inside_strike)
        j_inside_dips.append(j_inside_dip)
        
        #ax.pole(j_inside_strike, j_inside_dip, color=colours[joint])
    
    j_inside_strikes = j_inside_strikes[0] + j_inside_strikes[1] + j_inside_strikes[2]
    j_inside_dips = j_inside_dips[0] + j_inside_dips[1] + j_inside_dips[2]

    # Friction circle
    ax.cone(90,0,90-FrictionAngle, alpha=0.1, color='r', zorder=1, bidirectional=False)
    
    # Stereonet overlay settings
    ax.set_azimuth_ticks([0,90,180,270],labels=['N','E','S','W'])
    ax.grid(True, kind='polar')
    plt.tight_layout()
    
    # Shaded Area
    shaded_area = pole(j_inside_strikes, j_inside_dips)
    shaded_area_x = [x for x in shaded_area[0]]
    shaded_area_y = [x for x in shaded_area[1]]
    shaded_area_stack = np.vstack((shaded_area_x,shaded_area_y)).T
    
    edges = alpha_shape(shaded_area_stack, alpha=1, only_outer=True)
    edges_joined = stitch_boundaries(edges)
    edges_joined_x = []
    edges_joined_y = []

    for i, j in edges_joined[0]:
        edges_joined_x.append(shaded_area_x[i])
        edges_joined_y.append(shaded_area_y[i])
    polygon = np.vstack((edges_joined_x,edges_joined_y)).T

    ax.fill(edges_joined_x,edges_joined_y, 'k', alpha=0.3)
    
    Joints = ['1','2','3','1/2','1/3','2/3']
    
    Stability = "Stable"
    Mode = 'N/A'
    
    if point_inside_polygon(0, 0, polygon):
        Stability = 'Unstable'
        Mode = 'Falling'
    else:
        poles_ext = pole(point_strikes, [d+1 for d in point_dips])
        poles_ext = np.vstack((poles_ext[0],poles_ext[1])).T    
        max_slide = 0
        for i in range(0,6):
            if point_inside_polygon(poles_ext[i][0], poles_ext[i][1], polygon) and real_dips[i] >= FrictionAngle and real_dips[i] > max_slide:
                Stability = 'Unstable'
                max_slide = real_dips[i]
                Mode = "Sliding on Joint {0}".format(Joints[i])

    # Table in top plot
    table_data=[
        ["Friction Angle", "{0}°".format(FrictionAngle)],
        ["Joints", "Dip / Dip Direction"],
        ["1", "{0}°/{1:0=3d}°".format(int(round(Dip[0])),int(round(DipDirection[0])))],
        ["2", "{0}°/{1:0=3d}°".format(int(round(Dip[1])),int(round(DipDirection[1])))],
        ["3", "{0}°/{1:0=3d}°".format(int(round(Dip[2])),int(round(DipDirection[2])))],
        ["Intersections", "Trend / Plunge"],
        ["1/2", "{0}°/{1:0=3d}°".format(int(round(plunge[0])), int(round(bearing[0])))],
        ["1/3", "{0}°/{1:0=3d}°".format(int(round(plunge[1])), int(round(bearing[1])))],
        ["2/3", "{0}°/{1:0=3d}°".format(int(round(plunge[2])), int(round(bearing[2])))],
        ["Stability", Stability],
        ["Mode", Mode]
        ]
    
    table = ax.table(cellText=table_data, bbox = [1.1, 0.3, 0.4, 0.35], cellLoc='center', loc='right', colWidths=[0.2,0.2])# , loc='top right')#, colWidths = [0.4]*2)
    for (row, col), cell in table.get_celld().items():
        if (row == 0 or row ==1 or row == 5 or row == 9 or row == 10):
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
    html.H3(children='Kinematic Analysis of Underground Rock Wedge',
            style={'textAlign': 'center','font-family':'Verdana','color': colors['text'],'padding-top': 20}),
    html.Div([html.Label(["Joint 1 (Dip / Dip Direction)",dcc.Input(id='joint1_dip-state', type='number', min=1, max=89, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'}),dcc.Input(id='joint1_dd-state', type='number', min=0, max=359, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Label(["Joint 2 (Dip / Dip Direction)",dcc.Input(id='joint2_dip-state', type='number', min=1, max=89, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'}),dcc.Input(id='joint2_dd-state', type='number', min=0, max=359, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Label(["Joint 3 (Dip / Dip Direction)",dcc.Input(id='joint3_dip-state', type='number', min=1, max=89, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'}),dcc.Input(id='joint3_dd-state', type='number', min=0, max=359, style={'width': '80px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Label(["Friction Angle (°)",dcc.Input(id='fric_ang-state', type='number', min=1, max=89, value=30, style={'width': '50px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],
         style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([html.Button('Update Stereonet', id='update_button-state', n_clicks=0)],
             style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div(html.P([html.Br()])),
    html.Div(html.Img(id='graph',style={'width':'40%'}),style={'vertical-align':'middle', 'textAlign':'center'}),
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
        FrictionAngle = fric_ang
        
        # Stereonet on data
        stereofig, stereoax = stereoplot(Dip, DipDirection, FrictionAngle, (10,10))
        buf = io.BytesIO() # in-memory files
        stereofig.savefig(buf, format = "png", dpi=300, bbox_inches = "tight") # save to the above file object
        data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        plt.close()
        return "data:image/png;base64,{}".format(data)

if __name__ == "__main__":
    app.run_server()
    
                          
