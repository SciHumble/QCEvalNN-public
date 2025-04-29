import logging
import logging.config
import matplotlib


def setup_logging():
    matplotlib.set_loglevel("INFO")
    logger_lvl = "INFO"
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            },
        },
        'handlers': {
            'default': {
                'level': logger_lvl,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': logger_lvl,
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': 'qc_eval.log',
                'mode': 'a',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file'],
                'level': logger_lvl,
                'propagate': True
            },
            'qc_eval': {  # qc_eval package logger
                'handlers': ['default', 'file'],
                'level': logger_lvl,
                'propagate': False
            },
        }
    })


# Call setup_logging() to configure logging
setup_logging()
