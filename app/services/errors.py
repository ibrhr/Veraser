class VeraserError(Exception):
    pass


class SessionNotFoundError(VeraserError):
    pass


class UploadRejectedError(VeraserError):
    pass


class InvalidPromptError(VeraserError):
    pass


class ModelRuntimeError(VeraserError):
    pass


class MaskingJobNotFoundError(VeraserError):
    pass


class InpaintingJobNotFoundError(VeraserError):
    pass


class ObjectPromptNotFoundError(VeraserError):
    pass
