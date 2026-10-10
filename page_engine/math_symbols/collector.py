"""Local rectangular crop and explicit human label/appearance review."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from .library import Library, LABELS, prepared
from ..common import ROOT

class Collector:
    def __init__(self, root, library):
        self.root=root;self.library=library;self.source=None;self.page=None;self.rect=None;self.entry_id=None
        self.start=None;self.selection=None;self.photos=[]
        root.title('Handwriting AI — collect and REVIEW mathematical symbols')
        root.geometry('1250x850')
        toolbar=ttk.Frame(root);toolbar.pack(fill='x')
        ttk.Button(toolbar,text='Open scanned page',command=self.open_page).pack(side='left')
        self.page_name=tk.StringVar();ttk.Label(toolbar,textvariable=self.page_name).pack(side='left',padx=10)
        self.zoom=tk.DoubleVar(value=.35)
        ttk.Label(toolbar,text='Zoom').pack(side='left')
        z=ttk.Combobox(toolbar,textvariable=self.zoom,values=[.25,.35,.5,1,1.5,2],width=5,state='readonly');z.pack(side='left')
        z.bind('<<ComboboxSelected>>',lambda _:self.show_page())
        pane=ttk.Panedwindow(root,orient='horizontal');pane.pack(fill='both',expand=True)
        scan=ttk.Frame(pane);pane.add(scan,weight=3)
        self.canvas=tk.Canvas(scan,bg='#dddddd');self.canvas.grid(row=0,column=0,sticky='nsew')
        sx=ttk.Scrollbar(scan,orient='horizontal',command=self.canvas.xview);sx.grid(row=1,column=0,sticky='ew')
        sy=ttk.Scrollbar(scan,orient='vertical',command=self.canvas.yview);sy.grid(row=0,column=1,sticky='ns')
        self.canvas.configure(xscrollcommand=sx.set,yscrollcommand=sy.set)
        scan.rowconfigure(0,weight=1);scan.columnconfigure(0,weight=1)
        self.canvas.bind('<ButtonPress-1>',self.press);self.canvas.bind('<B1-Motion>',self.drag);self.canvas.bind('<ButtonRelease-1>',self.release)
        right=ttk.Frame(pane,padding=8);pane.add(right,weight=1)
        ttk.Label(right,text='Drag a tight rectangle at 100% zoom.\nNo label is inferred from the image.').pack(anchor='w')
        self.preview=ttk.Label(right);self.preview.pack(pady=8)
        self.label=tk.StringVar();self.cutoff=tk.IntVar(value=220);self.baseline=tk.DoubleVar(value=.9);self.height=tk.DoubleVar(value=1)
        fields=ttk.Frame(right);fields.pack(fill='x')
        for row,(title,var,values) in enumerate([('Exact symbol',self.label,LABELS),('Ink cutoff (80–250)',self.cutoff,None),
               ('Baseline fraction (0–1)',self.baseline,None),('Height / prose x-height',self.height,None)]):
            ttk.Label(fields,text=title).grid(row=row,column=0,sticky='w')
            if values:widget=ttk.Combobox(fields,textvariable=var,values=values,width=16)
            else:widget=ttk.Entry(fields,textvariable=var,width=16)
            widget.grid(row=row,column=1,sticky='ew');widget.bind('<KeyRelease>',self.changed)
            if values:widget.bind('<<ComboboxSelected>>',self.changed)
        ttk.Button(right,text='Update ink/baseline preview',command=self.show_crop).pack(fill='x',pady=4)
        ttk.Label(right,text='Red line = baseline. Adjust before approval.\nBody letters: ~1; capitals/digits: ~1.4; bars: ~0.1.\nCrop faint grid lines out; change cutoff carefully.\nRAW crop is preserved; preview uses processed ink.').pack(anchor='w')
        self.reviewed=tk.BooleanVar(value=False)
        for variable in (self.label,self.cutoff,self.baseline,self.height):
            variable.trace_add('write',lambda *_:self.changed())
        ttk.Checkbutton(right,text='I checked BOTH the symbol label and crop',variable=self.reviewed).pack(anchor='w',pady=8)
        ttk.Button(right,text='Save draft (not usable yet)',command=lambda:self.save(False)).pack(fill='x')
        ttk.Button(right,text='Approve checked crop and label',command=lambda:self.save(True)).pack(fill='x',pady=4)
        ttk.Label(right,text='Saved variants — select to review/edit:\nEditing an approved entry clears its approval.').pack(anchor='w')
        self.entries=tk.Listbox(right,height=13,exportselection=False);self.entries.pack(fill='both',expand=True)
        self.entries.bind('<<ListboxSelect>>',self.select_entry)
        ttk.Button(right,text='New variant (draw another rectangle)',command=self.reset).pack(fill='x')
        self.status=tk.StringVar(value='Select your page and draw a crop. Nothing is approved automatically.')
        ttk.Label(root,textvariable=self.status,wraplength=1200).pack(fill='x',padx=8,pady=5)
        self.refresh()
        pages=sorted((ROOT/'data/pages').glob('*.png'))
        if pages:self.open_page(pages[0])

    def open_page(self,path=None):
        path=path or filedialog.askopenfilename(initialdir=ROOT/'data/pages',filetypes=[('Scanned image','*.png *.jpg *.jpeg *.tif *.tiff')])
        if not path:return
        try:
            with Image.open(path) as image:self.page=image.convert('RGB')
            self.source=Path(path).resolve();self.page_name.set(self.source.name);self.reset();self.show_page()
        except Exception as error:messagebox.showerror('Cannot open page',str(error))

    def show_page(self):
        if self.page is None:return
        zoom=self.zoom.get();image=self.page.resize((round(self.page.width*zoom),round(self.page.height*zoom)),Image.Resampling.LANCZOS)
        self.page_photo=ImageTk.PhotoImage(image);self.canvas.delete('all');self.selection=None
        self.canvas.create_image(0,0,image=self.page_photo,anchor='nw');self.canvas.configure(scrollregion=(0,0,image.width,image.height))

    def press(self,event):
        if self.page is None:return
        self.start=(self.canvas.canvasx(event.x),self.canvas.canvasy(event.y))
        if self.selection:self.canvas.delete(self.selection)
        self.selection=self.canvas.create_rectangle(*self.start,*self.start,outline='red',width=2)

    def drag(self,event):
        if self.start:self.canvas.coords(self.selection,*self.start,self.canvas.canvasx(event.x),self.canvas.canvasy(event.y))

    def release(self,event):
        if self.start is None:return
        x,y=self.start;r,b=self.canvas.canvasx(event.x),self.canvas.canvasy(event.y);z=self.zoom.get()
        self.rect=(max(0,round(min(x,r)/z)),max(0,round(min(y,b)/z)),min(self.page.width,round(max(x,r)/z)),min(self.page.height,round(max(y,b)/z)))
        self.start=None;self.entry_id=None;self.label.set('');self.reviewed.set(False)
        if self.rect[2]<=self.rect[0] or self.rect[3]<=self.rect[1]:self.rect=None;return
        self.raw=self.page.crop(self.rect);self.show_crop();self.status.set('New unlabelled crop. Enter its exact label and review before approval.')

    def changed(self,event=None):self.reviewed.set(False)

    def show_crop(self):
        if not hasattr(self,'raw'):return
        try:
            image,_=prepared(self.raw,self.cutoff.get());scale=min(3,350/image.width,180/image.height)
            image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS).convert('RGB')
            y=round(self.baseline.get()*(image.height-1));ImageDraw.Draw(image).line((0,y,image.width,y),fill='red')
            raw=self.raw.copy();raw.thumbnail((350,130))
            panel=Image.new('RGB',(max(raw.width,image.width),raw.height+image.height+24),'white')
            panel.paste(raw,(0,0));panel.paste(image,(0,raw.height+24))
            self.crop_photo=ImageTk.PhotoImage(panel);self.preview.configure(image=self.crop_photo)
        except Exception as error:self.status.set(str(error))

    def save(self,approve):
        try:
            if not hasattr(self,'raw'):raise ValueError('Draw or select a crop first')
            if approve and not self.reviewed.get():raise ValueError('Review the label AND prepared crop, then check the review box')
            values=(self.label.get(),self.cutoff.get(),self.baseline.get(),self.height.get())
            if self.entry_id is None:
                entry=self.library.capture(self.source,self.rect,*values);self.entry_id=entry['id']
            entry=self.library.update(self.entry_id,*values,approve=approve,reviewed=self.reviewed.get())
            self.refresh();self.status.set(f"Saved {entry['label']} — {entry['status']}; originals untouched.")
        except Exception as error:messagebox.showerror('Symbol not saved',str(error))

    def refresh(self):
        self.entries.delete(0,'end')
        for entry in self.library.data['entries']:self.entries.insert('end',f"{entry['label']} | {entry['status']} | {entry['id'][:8]}")

    def select_entry(self,event=None):
        selected=self.entries.curselection()
        if not selected:return
        entry=self.library.data['entries'][selected[0]];self.entry_id=entry['id']
        with Image.open(self.library.directory/entry['image']) as image:self.raw=image.copy()
        self.label.set(entry['label']);self.cutoff.set(entry['cutoff']);self.baseline.set(entry['baseline']);self.height.set(entry['height_em'])
        self.reviewed.set(False);self.show_crop();self.status.set('Review saved variant. Save draft revokes prior approval; raw crop is never overwritten.')

    def reset(self):
        self.entry_id=None;self.rect=None;self.label.set('');self.reviewed.set(False)
        if hasattr(self,'raw'):del self.raw
        self.preview.configure(image='')

def launch(library):
    root=tk.Tk();Collector(root,library);root.mainloop()
