import json
import argparse
from app import main
from app.settings import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    with open(config.APP_PATH_PARAMS_JSON_FILE, 'r', encoding='utf-8') as json_file:
        params = json.load(json_file)

    main.execute(params, debug_mode=args.debug)
