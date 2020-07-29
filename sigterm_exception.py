class SigTermException(Exception):
    def __init__(self, message,errors):
        super().__init__(message)
        self.errors = errors

def signal_handler(signum, frame):
        raise SigTermException("SIGTERM","SIGTERM sent from OS.")
