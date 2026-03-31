import tkinter as tk
from tkinter import scrolledtext
import subprocess # to run other scripts
import os
import json # save window size for later

# config file name
CONFIG_FILE = "gui_config.json"

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("eScribe GUI Test")
        
        # load save/defaults
        self.load_settings()
        self.root.configure(bg='black')
        # var for background process
        self.proc_receiver = None
        self.proc_whisper = None
        #default font size
        self.font_size = 18
        self.last_size = 0
        # change this file for grabbing the text
        self.live_file = "live_transcript.txt" 

        # bind resize to save
        self.root.bind("<Configure>", self.save_settings_trigger)

        self.create_widgets()
        self.monitor_file()

    def create_widgets(self):
        # header properties
        header = tk.Frame(self.root, bg='black', height=50)
        header.pack(side="top", fill="x")
        
        # exit button properties
        tk.Button(header, text="X", command=self.emergency_stop, 
                  bg="pink", fg="black").pack(side="left", padx=0)

        # footer properties
        controls = tk.Frame(self.root, bg='black')
        controls.pack(side="bottom", fill="x", pady=10)

        # start receiving button properties
        self.recv_btn = tk.Button(controls, text="Start Receiver", command=self.toggle_receiver, width=15)
        self.recv_btn.pack(side="left", padx=10)

        # start transcribing button properties
        self.trans_btn = tk.Button(controls, text="Start Transcribing", command=self.toggle_transcribing, width=15)
        self.trans_btn.pack(side="left", padx=10)

        # font size button properites
        tk.Button(controls, text="Size+", command=lambda: self.adjust_font(2)).pack(side="right", padx=10)
        tk.Button(controls, text="Size-", command=lambda: self.adjust_font(-2)).pack(side="right", padx=10)

        # main display properties
        self.display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Arial", self.font_size),
        bg="black", fg="white")
        self.display.pack(side="top", expand=True, fill="both", padx=20, pady=5)

    # --- Script Control Logic ---

    def toggle_receiver(self):
        if self.proc_receiver is None:
            self.proc_receiver = subprocess.Popen(['python3', 'Escribe Receiver.py'])
            self.recv_btn.config(text="Stop Receiving", bg="orange")
        else:
            self.proc_receiver.terminate()
            self.proc_receiver = None
            self.recv_btn.config(text="Start Receiving", bg="SystemButtonFace")

    def toggle_transcribing(self):
        if self.proc_whisper is None:
            self.proc_whisper = subprocess.Popen(['python3', 'whisper_engine.py'])
            self.trans_btn.config(text="Stop Transcribing", bg="green")
        else:
            self.proc_whisper.terminate()
            self.proc_whisper = None
            self.trans_btn.config(text="Start Transcribing", bg="SystemButtonFace")

    def emergency_stop(self):
        """Kills all sub-processes and exits"""
        if self.proc_receiver: self.proc_receiver.terminate()
        if self.proc_whisper: self.proc_whisper.terminate()
        self.root.quit()

    # --- LAYOUT PERSISTENCE LOGIC ---

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.root.geometry(f"{data['w']}x{data['h']}+{data['x']}+{data['y']}")
        else:
            self.root.geometry("800x600+100+100")

    def save_settings_trigger(self, event):
        if event.widget == self.root:
            geometry = self.root.geometry() 
            w, rest = geometry.split('x')
            h, x, y = rest.replace('+', ' ').replace('-', ' ').split()
            settings = {"w": w, "h": h, "x": x, "y": y}
            with open(CONFIG_FILE, "w") as f:
                json.dump(settings, f)

    def adjust_font(self, delta):
        self.font_size += delta
        self.display.configure(font=("Arial", self.font_size))

    def monitor_file(self):
        if os.path.exists(self.live_file):
            size = os.path.getsize(self.live_file)
            if size > self.last_size:
                with open(self.live_file, "r") as f:
                    self.display.delete('1.0', tk.END)
                    self.display.insert(tk.END, f.read())
                    self.display.see(tk.END)
                self.last_size = size
        self.root.after(500, self.monitor_file)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()