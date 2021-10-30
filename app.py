from webapp import app

import argparse
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
