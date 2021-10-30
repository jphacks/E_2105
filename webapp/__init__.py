import dash
import dash_bootstrap_components as dbc
import logging
import sys


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP,
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

import webapp.dash
import webapp.event_handler
