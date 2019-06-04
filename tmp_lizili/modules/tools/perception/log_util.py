import logging
from logging.config import dictConfig

logging_config = {
    'version': 1,
    'formatters': {
        'f': {
            'format': '%(asctime)s - %(levelname)s - line %(lineno)d => %(message)s'
        }
    },

    'handlers': {
        'h1': {
            'class': 'logging.StreamHandler',
            'formatter': 'f',
        },
    },
    'root': {
        'handlers': ['h1'],
        'level': logging.INFO
    }
}

dictConfig(logging_config)
logger = logging.getLogger()
