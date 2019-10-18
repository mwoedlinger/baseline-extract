class Point:
    """
    Represents a point in 2 dim space.
    """
    def __init__(self, x: int = 0, y: int = 0):
        self.c = [x, y]

    def scale(self, scale_factor_x: float, scale_factor_y: float):
        """
        Scales both coordinates with the given scale_factor
        """
        self.c[0] *= scale_factor_x
        self.c[1] *= scale_factor_y

    def set_from_string(self, coords: str, sep: str = ','):
        """
        Sets the coordinates according to the String coords assuming the strucutre: 'xsepy'.
        Example: coords = '13,12' amd sep = ','
        :param coords: Coordinate String
        :param sep: Seperator
        """
        self.c[0] = int(coords.split(sep)[0])
        self.c[1] = int(coords.split(sep)[1])

    def get_as_list(self) -> list:
        """
        Returns the coordinates as a two dimensional list
        """
        return self.c.copy()

    def __str__(self):
        return str(round(self.c[0])) + ',' + str(round(self.c[1]))