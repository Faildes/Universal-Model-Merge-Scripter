import os
import pickle
import tkinter as tk
from tkinter import filedialog
from plan import create_plan, create_plan_ipynb

config_file = os.path.join(os.getcwd(),"config.tccm")

def save_set(config):
    if os.path.exists(config_file):
        os.remove(config_file)
    pickle.dump(config, open(config_file, "wb"))
    
def load_set():
    return pickle.load(open(config_file,"rb"))

init_config={
    "filepath":os.path.dirname(__file__),
    "saveas":os.path.dirname(__file__),
    "title":"",
    "vae":"",
    "CivitAPI":"Please Insert CivitAI API",
    "HuggingAPI":"Please Insert HuggingFace write API",
    "UR":"User/Repo"}
if os.path.exists(config_file):
    config=load_set()
else:
    config=init_config

# ファイル指定の関数
def filedialog_clicked():
    fTyp = [("Text File","*.txt")]
    iFile = cp("filepath")
    iFilePath = filedialog.askopenfilename(filetype = fTyp, title = "Select Planned File Path", initialdir = iFile)
    fileentry.delete(0, tk.END)
    fileentry.insert(tk.END, iFilePath)
    config["filepath"] = iFilePath

def about():
    sub_win = tk.Toplevel()
    sub_win.geometry("300x100")
    label_sub = tk.Label(sub_win, text="Model Planner\nMade by Team C 2024")
    label_sub.pack()

def cp(name):
    try:
        return config[name]
    except:
        return init_config[name]

def main():
    win=tk.Tk()
    menu=tk.Menu(win)
    win.config(menu=menu)
    menu_file = tk.Menu(win)
    menu.add_cascade(label='Help', menu=menu_file)
    menu_file.add_command(label='About', command=about)

    win.geometry("330x270+100+120")
    win.resizable(True, True)
    win.title("Model Planner")

    frame1 = tk.Frame(win)
    frame2 = tk.Frame(win)
    frame3 = tk.Frame(win)
    frame4 = tk.Frame(win)
    frame5 = tk.Frame(win)
    frame6 = tk.Frame(win)
    frame7 = tk.Frame(win)

    newlab = tk.Label(win, text="Model Planner", font = ('MS Gothic', 20))
    lab = tk.Label(frame1, text="Planned Text Path", font = ('MS Gothic', 10))
    lab1 = tk.Label(frame2, text="Title", font = ('MS Gothic', 10))
    lab2 = tk.Label(frame3, text="VAE Link", font = ('MS Gothic', 10))
    lab3 = tk.Label(frame4, text="CivitAI API", font = ('MS Gothic', 10))
    lab4 = tk.Label(frame5, text="HuggingFace API", font = ('MS Gothic', 10))
    lab5 = tk.Label(frame6, text="User/Repo", font = ('MS Gothic', 10))
    fileentry = tk.Entry(frame1)
    fileentry.insert(0,cp("filepath"))
    def filedialog_clicked():
        fTyp = [("Text File","*.txt")]
        iFile = cp("filepath")
        iFilePath = filedialog.askopenfilename(filetype = fTyp, title = "Select Planned File Path", initialdir = iFile)
        fileentry.delete(0, tk.END)
        fileentry.insert(tk.END, iFilePath)
        config["filepath"] = iFilePath
    filebutton = tk.Button(frame1, text="Open",command=filedialog_clicked)
    title = tk.Entry(frame2,width=20)
    title.insert(0,cp("title"))
    vae = tk.Entry(frame3,width=20)
    vae.insert(0,cp("vae"))
    CAPI = tk.Entry(frame4,width=20)
    CAPI.insert(0,cp("CivitAPI"))
    HAPI = tk.Entry(frame5,width=20)
    HAPI.insert(0,cp("HuggingAPI"))
    UR = tk.Entry(frame6,width=20)
    UR.insert(0,cp("UR"))
    def save_as_text():
        config["title"] = title.get()
        config["vae"] = vae.get()
        config["CivitAPI"] = CAPI.get()
        config["HuggingAPI"] = HAPI.get()
        fTyp = [("Text File","*.txt")]
        iFile = cp("saveas")
        filename = filedialog.asksaveasfilename(initialdir = iFile,title = "Save as",filetypes =  fTyp)
        if not filename.endswith(".txt"):
            filename += ".txt"
        config["saveas"] = filename
        if os.path.exists(filename):
            os.remove(filename)
        create_plan(**config)
        save_set(config)

    def save_as_ipynb():
        config["title"] = title.get()
        config["vae"] = vae.get()
        config["CivitAPI"] = CAPI.get()
        config["HuggingAPI"] = HAPI.get()
        config["UR"] = UR.get()
        fTyp = [("IPYNB File","*.ipynb")]
        iFile = cp("saveas")
        filename = filedialog.asksaveasfilename(initialdir = iFile,title = "Save as",filetypes =  fTyp)
        if not filename.endswith(".ipynb"):
            filename += ".ipynb"
        config["saveas"] = filename
        if os.path.exists(filename):
            os.remove(filename)
        create_plan_ipynb(**config)
        save_set(config)
    save = tk.Button(frame7, text="Save Plan As Text",command=save_as_text)
    save1 = tk.Button(frame7, text="Save Plan As .ipynb",command=save_as_ipynb)

    newlab.pack(anchor=tk.NW)
    frame1.pack(anchor=tk.NW,expand=True,fill="x",pady=(10,0))
    lab.pack(anchor=tk.NW)
    fileentry.pack(side=tk.LEFT,expand=True,fill="x",padx=(0,10))
    filebutton.pack(side=tk.LEFT)
    frame2.pack(anchor=tk.NW,expand=True,fill="x",pady=(10,0))
    lab1.pack(side=tk.LEFT)
    title.pack(side=tk.LEFT,expand=True,fill="x",padx=(0,10))
    frame3.pack(anchor=tk.NW,expand=True,fill="x",pady=(10,0))
    lab2.pack(side=tk.LEFT)
    vae.pack(side=tk.LEFT,expand=True,fill="x",padx=(0,10))
    frame4.pack(anchor=tk.NW,expand=True,fill="x",pady=(10,0))
    lab3.pack(side=tk.LEFT)
    CAPI.pack(side=tk.LEFT,expand=True,fill="x",padx=(0,10))
    frame5.pack(anchor=tk.NW,expand=True,fill="x",pady=(10,0))
    lab4.pack(side=tk.LEFT)
    HAPI.pack(side=tk.LEFT,expand=True,fill="x",padx=(0,10))
    frame6.pack(anchor=tk.NW,expand=True,fill="x",pady=(10,0))
    lab5.pack(side=tk.LEFT)
    UR.pack(side=tk.LEFT,expand=True,fill="x",padx=(0,10))
    frame7.pack(anchor=tk.S,expand=True,fill="x")
    save.pack(side=tk.LEFT)
    save1.pack(side=tk.RIGHT)
    win.mainloop()

if __name__ == "__main__":
    main()
