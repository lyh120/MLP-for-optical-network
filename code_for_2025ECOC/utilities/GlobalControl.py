import logging
import time
import coloredlogs


class GlobalControl(object):
    # logger = None
    def __init__(cls, logger=None):
        cls.logger = logger
    
    @classmethod
    def init_logger(cls, name='logger' + time.strftime('%Y-%m-%d',time.localtime(time.time())), level=1, color_set_mode='modified'):
        cls.logger = MyLogging(name, level, color_set_mode).logger
        return cls.logger
    

class MyLogging(object):
    logger = logging.Logger('default')
    def __init__(self, name, level=1, color_set_mode='modified'):
        self.logger = logging.Logger(name)
        fmt = '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s[line:%(lineno)d] %(message)s'
        formater = logging.Formatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formater)
        self.logger.addHandler(ch)
        if color_set_mode=='modified':
            coloredlogs.DEFAULT_FIELD_STYLES = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'},
                                                'levelname': {'color': 'green', 'bold': True}, 'request_id':{'color': 'yellow'},
                                                'name': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'threadName': {'color': 'yellow'}}
        coloredlogs.install(fmt=fmt, level=level, logger=self.logger)


if __name__ == '__main__':
    gc = GlobalControl.init_logger()
    gc.logger.debug('test debug')
    gc.logger.info('test info')
    gc.logger.warning('test warning')
