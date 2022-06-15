import pyglet


class SimpleImageViewer(object):

    def __init__(self, display=None, caption="THOR Browser", name=""):
        self.window = None
        self.isopen = False
        self.display = display
        self.caption = caption
        self.iter = 0
        self.name = name

    def reset(self):
        self.iter = 0

    def imshow(self, arr, save=False):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display, caption=self.caption)
            self.width = width
            self.height = height
            self.isopen = True

        assert arr.shape == (
            self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()
        if save:
            filename = 'images/' + self.name + \
                format(self.iter, '05d') + '.jpg'
            self.iter = self.iter + 1
            image.save(filename)

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
