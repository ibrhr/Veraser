class VeraserError(Exception):
    pass


class SessionNotFoundError(VeraserError):
    pass


class UploadRejectedError(VeraserError):
    pass


class InvalidPromptError(VeraserError):
    pass
