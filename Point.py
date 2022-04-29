def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        """
        The Point class is used to define any arbitrary point on our observation image
        Any class inheriting this must provide "icon_h" and "icon_w" representing the
        height and width of the icon associated with the element.

        :param name: Name of the point ("Bird", "Chopper", "Fuel")
        :param x_max: Maximum permissible x coordinate, value is clamped if provided x larger than this.
        :param x_min: Minimum permissible x coordinate, value is clamped if provide x smaller than this.
        :param y_max: Maximum permissible y coordinate, value is clamped if provided y larger than this.
        :param y_min: Minimum permissible y coordinate, value is clamped if provide y smaller than this.
        """

        # (x, y) Position of the point on the image

        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

        self.icon_h = None
        self.icon_w = None

    def set_position(self, x, y):
        """
        Sets the points new position. The coordinate values are clamped based on limiters defined during
        initialization of the class

        :param x: new x coordinate of the point
        :param y: new y coordinate of the point
        :return:
        """

        self.x = clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        """
        Getter for current coordinate on the image screen

        :return: (int, int) current position of the point
        """
        return self.x, self.y

    def move(self, del_x, del_y):
        """
        Moves the point by provided deltas.

        :param del_x: change in x coordinate
        :param del_y: change in y coordinate
        :return:
        """
        self.x += del_x
        self.y += del_y

        self.x = clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = clamp(self.y, self.y_min, self.y_max - self.icon_h)


