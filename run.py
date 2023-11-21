import json
from app import main
from app.settings import config


if __name__ == '__main__':
    with open(config.APP_PATH_PARAMS_JSON_FILE, 'r', encoding='utf-8') as json_file:
        params = json.load(json_file)

    main.execute(params)
