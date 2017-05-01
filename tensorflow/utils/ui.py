from Tkinter import *


class CanvasDialog:
    
    def __init__(self, title, width, height, thickness=20, resultScale=10):
        self.title = title
        self.width = width
        self.height = height
        self.thickness = thickness
        self.resultScale = resultScale
        
        self.root = Tk()
        self.root.title(title)
        self.root.resizable(0,0)
        self.root.protocol("WM_DELETE_WINDOW", self._closing)

        self.canvas = Canvas(self.root, bg="white", width=width, height=height)
        self.canvas.configure(cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._mouse_left)
        self.canvas.bind("<Button-3>", self._mouse_right)
        self.canvas.bind("<B1-Motion>", self._mouse_move)
        self.resultsData = []
        

    def show(self):
            self.root.mainloop()
            return self.resultsData

    def _get_pixel(self, x, y):
        ids = self.canvas.find_overlapping(x, y, x, y)
        if len(ids) > 0:
            return 1.0
        return 0.0

    def _get_pixel_group(self, x, y, width, height):
        total = width * height
        pixCounter = 0.0
        for r in range(y, y+height):
            for c in range(x, x+width):
                if self._get_pixel(c, r) > 0:
                    pixCounter += 1.0
        return pixCounter / total
        
    def _get_data(self):
        counter = 0
        scaledImg = []
        for y in range(0, self.height, self.resultScale):
            for x in range(0, self.width, self.resultScale):
                scaledImg.append(self._get_pixel_group(x, y,
                                                       self.resultScale,
                                                       self.resultScale));
        return scaledImg

    def _mouse_left(self, event):
        print(self.get_pixel(event.x, event.y))
        self.canvas.create_oval(event.x - self.thickness / 2,
                                event.y - self.thickness / 2,
                                event.x + self.thickness / 2,
                                event.y + self.thickness / 2,
                                fill="black")

    def _mouse_move(self ,event):
        self.canvas.create_oval(event.x - self.thickness / 2,
                                event.y - self.thickness / 2,
                                event.x + self.thickness / 2,
                                event.y + self.thickness / 2,
                                fill="black")	

    def _mouse_right(self, event):
        self.canvas.delete("all")
        
    def _closing(self):
        self.resultsData = self._get_data()
        self.root.destroy()
        