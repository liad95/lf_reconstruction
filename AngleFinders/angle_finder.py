from abc import abstractmethod


class angle_finder:
    def __init__(self):
        pass

    @abstractmethod
    def get_delta_sin(self, mask, filter=None):
        pass
