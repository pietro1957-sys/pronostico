import os
import sys
import threading
import time
import webbrowser
from streamlit.web import cli as stcli


def open_browser():
  time.sleep(4)
  webbrowser.open('http://localhost:8501')


def main():
  if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
  else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

  script_path = os.path.join(base_dir, 'pronostico.py')

  threading.Thread(target=open_browser, daemon=True).start()

  sys.argv = [
      'streamlit',
      'run',
      script_path,
      '--global.developmentMode=false',
      '--server.headless=true',
      '--server.port=8501',
  ]

  sys.exit(stcli.main())


if __name__ == '__main__':
  main()