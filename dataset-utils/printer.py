import math
from IPython.display import display

class DHPrinter():
    class Printer(str):
        def __repr__(self):
            return self
        
    def __init__(self):
        self.dh = None
        
    def print(self, content):
        if self.dh is None:
            self.dh = display(self.Printer(content), display_id=True)
        else:
            self.dh.update(self.Printer(content))

    def print_progress(self, content, perc_complete):
        t = 20
        a = math.floor(t * perc_complete)
        b = t - a
        
        complete = "=" * a
        remaining = "-" * b
        progress_bar = f"|{complete}{">" if a != t else ""}{remaining}|{perc_complete:.1%}"
        self.print(f"{content}\n{progress_bar}")