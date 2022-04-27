def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return self.x, self.y

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = clamp(self.y, self.y_min, self.y_max - self.icon_h)


