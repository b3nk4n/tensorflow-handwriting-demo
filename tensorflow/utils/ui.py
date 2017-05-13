from Tkinter import *


class CanvasDialog:
    
    def __init__(self, title, width, height, scale=10, num_letters=1):
        self.resultsData = []
        self.title = title
        self.scaled_width = width*scale
        self.scaled_height = height*scale
        self.scale = scale
        self.num_letters = num_letters
        
        self.root = Tk()
        self.root.title(title)
        self.root.resizable(0,0)
        self.root.protocol("WM_DELETE_WINDOW", self._closing)

        self.canvas = Canvas(self.root, bg="white",
                             width=self.scaled_width*num_letters+(self.num_letters-1),
                             height=self.scaled_height)
        self.canvas.configure(cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._mouse_left)
        self.canvas.bind("<Button-3>", self._mouse_right)
        self.canvas.bind("<B1-Motion>", self._mouse_move)
        self._render_seperators()
      
    def show(self):
        self.root.mainloop()
        return self.resultsData
        
    def _render_seperators(self):
        for i in range(1, self.num_letters):
            x = i*self.scaled_width + (i-1)*1
            self.canvas.create_line(x, 0,
                                    x, self.scaled_height,
                                    fill="grey", dash=(4,4)) 

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
        dataList = []
        for i in range(self.num_letters):
            scaledData = []
            isEmptyLetter = True
            for y in range(0, self.scaled_width, self.scale):
                for x in range(i*(self.scaled_width+1),
                               i*(self.scaled_width+1)+self.scaled_width,
                               self.scale):
                    group = self._get_pixel_group(x,
                                                  y,
                                                  self.scale,
                                                  self.scale)                              
                    scaledData.append(group)
                    if group > 0:
                        isEmptyLetter = False
            if not isEmptyLetter: 
                dataList.append(scaledData)
        return dataList

    def _mouse_left(self, event):
        thickness = self.scale * 4
        self.canvas.create_oval(event.x - thickness / 2,
                                event.y - thickness / 2,
                                event.x + thickness / 2,
                                event.y + thickness / 2,
                                fill="black")

    def _mouse_move(self ,event):
        thickness = self.scale * 4
        self.canvas.create_oval(event.x - thickness / 2,
                                event.y - thickness / 2,
                                event.x + thickness / 2,
                                event.y + thickness / 2,
                                fill="black")	

    def _mouse_right(self, event):
        self.canvas.delete("all")
        self._render_seperators()
        
    def _closing(self):
        self.resultsData = self._get_data()
        self.root.destroy()
        