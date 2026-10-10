"""Small Tkinter word reviewer. Tk calls stay on the main thread."""
import queue
import hashlib
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from .common import ROOT,read_json,write_json

class Reviewer:
    def __init__(self, root, session, store):
        self.root,self.session,self.store = root,session,store
        self.index = 0; self.offset = 0; self.busy = False; self.photos=[]; self.jobs=queue.Queue()
        session_id=hashlib.sha256(json.dumps([session['model'],session['words']],sort_keys=True).encode()).hexdigest()[:24]
        self.cursor_path=ROOT/'.reforge-cache/review_positions'/f'{session_id}.json'
        if self.cursor_path.exists():
            cursor=read_json(self.cursor_path)
            self.index=max(0,min(len(session['words'])-1,cursor['index']))
        root.title('Handwriting AI — word candidate review (Step 9.5)')
        root.geometry('1120x530'); root.minsize(900,500)
        self.title = ttk.Label(root,font=('Arial',18)); self.title.pack(pady=10)
        self.status = ttk.Label(root,wraplength=1050); self.status.pack()
        self.cards = ttk.Frame(root); self.cards.pack(fill='both',expand=True,padx=12,pady=12)
        self.card_widgets=[]
        for c in range(3):
            frame=ttk.Frame(self.cards,relief='ridge',padding=8); frame.grid(row=0,column=c,sticky='nsew',padx=4)
            self.cards.columnconfigure(c,weight=1)
            label=ttk.Label(frame); label.pack(pady=8)
            detail=ttk.Label(frame,wraplength=310); detail.pack(pady=8)
            button=ttk.Button(frame,text='Choose and approve',command=lambda c=c:self.choose(self.offset+c)); button.pack(pady=8)
            self.card_widgets.append((label,detail,button))
        note_frame=ttk.Frame(root); note_frame.pack(fill='x',padx=20)
        ttk.Label(note_frame,text='Problem note:').pack(side='left')
        self.note=tk.StringVar(); ttk.Entry(note_frame,textvariable=self.note).pack(side='left',fill='x',expand=True)
        nav=ttk.Frame(root); nav.pack(pady=8)
        self.buttons=[]
        for label, command in [('Previous word',lambda:self.move(-1)),('Next word',lambda:self.move(1)),
                               ('Next unreviewed',self.next_pending),('Earlier candidates',lambda:self.page(-3)),
                               ('Later candidates',lambda:self.page(3)),('Mark problematic',self.problem),('Set pending',self.pending)]:
            b=ttk.Button(nav,text=label,command=command); b.pack(side='left',padx=2); self.buttons.append(b)
        generation=ttk.Frame(root);generation.pack(pady=8)
        ttk.Label(generation,text='New seed (blank = next deterministic seed):').pack(side='left')
        self.seed=tk.StringVar(); ttk.Entry(generation,textvariable=self.seed,width=15).pack(side='left',padx=6)
        self.regen=ttk.Button(generation,text='Generate another candidate',command=self.regenerate);self.regen.pack(side='left')
        ttk.Label(root,text='Decisions save immediately. Approval is your visual judgment, not an OCR guarantee. Re-render via the quality CLI.').pack(pady=8)
        self.poll_id=None
        root.bind('<Destroy>',self.on_destroy,add='+')
        self.refresh(); self.poll_id=root.after(100,self.poll)

    def on_destroy(self,event):
        if event.widget==self.root and self.poll_id:
            try:self.root.after_cancel(self.poll_id)
            except tk.TclError:pass
            self.poll_id=None

    def item(self):return self.session['words'][self.index]
    def bundle(self):return self.store.load(self.item()['bundle_id'])
    def refresh(self):
        item=self.item()
        self.title.configure(text=f'Word {self.index+1}/{len(self.session["words"])}:  {item["text"]}')
        try: bundle=self.bundle()
        except FileNotFoundError:
            self.status.configure(text='Candidates not prepared yet. Finish the prepare command, then reopen review.');return
        self.note.set(bundle['problem_note'])
        selected=bundle['selected']
        self.status.configure(text=f'Status: {bundle["status"]} | Selected: {selected+1 if selected is not None else "none (preview uses first available)"} | {len(bundle["candidates"])} candidates. Raw images below; no spelling is assumed.')
        self.photos=[]
        for column,(label,detail,button) in enumerate(self.card_widgets):
            n=self.offset+column
            if n>=len(bundle['candidates']):
                label.configure(image='');detail.configure(text='No candidate');button.configure(state='disabled');continue
            candidate=bundle['candidates'][n]
            image=Image.open(ROOT/candidate['image']).convert('RGB')
            scale=min(310/image.width,120/image.height)
            image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
            photo=ImageTk.PhotoImage(image,master=self.root);self.photos.append(photo);label.configure(image=photo)
            label.bind('<Double-Button-1>',lambda event,path=candidate['image']:self.zoom(path))
            detail.configure(text=f'Candidate {n+1} | seed {candidate["seed"]}\n'+(', '.join(candidate['flags']) or 'No structural flag; text unverified')+('\nSELECTED' if n==selected else ''))
            button.configure(state='disabled' if self.busy else 'normal')

    def choose(self, selected):
        if self.busy:return
        self.store.decide(self.item()['bundle_id'],selected,'approved',self.note.get());self.refresh()
    def zoom(self,path):
        window=tk.Toplevel(self.root);window.title('Raw generated candidate — inspect letter details')
        image=Image.open(ROOT/path).convert('RGB');scale=min(3,1000/image.width)
        image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
        window.photo=ImageTk.PhotoImage(image,master=window);ttk.Label(window,image=window.photo).pack(padx=10,pady=10)
    def save_cursor(self):write_json(self.cursor_path,{'index':self.index})
    def problem(self):
        if not self.busy:self.store.decide(self.item()['bundle_id'],status='problematic',note=self.note.get());self.refresh()
    def pending(self):
        if not self.busy:self.store.decide(self.item()['bundle_id'],status='pending',note=self.note.get());self.refresh()
    def move(self, amount):
        if self.busy:return
        self.index=(self.index+amount)%len(self.session['words']);self.offset=0;self.save_cursor();self.refresh()
    def page(self, amount):
        if self.busy:return
        maximum=max(0,((len(self.bundle()['candidates'])-1)//3)*3)
        self.offset=max(0,min(maximum,self.offset+amount));self.refresh()
    def next_pending(self):
        if self.busy:return
        for delta in range(1,len(self.session['words'])+1):
            index=(self.index+delta)%len(self.session['words'])
            if self.store.load(self.session['words'][index]['bundle_id'])['status']!='approved':
                self.index=index;self.offset=0;self.save_cursor();self.refresh();return
    def regenerate(self):
        if self.busy:return
        try:seed=int(self.seed.get()) if self.seed.get().strip() else None
        except ValueError:messagebox.showerror('Seed','Enter an integer seed');return
        if seed is not None and not 0<=seed<2**63:messagebox.showerror('Seed','Seed must be between 0 and 2^63-1');return
        self.busy=True;self.regen.configure(state='disabled');self.refresh()
        self.status.configure(text='Generating one new candidate. Existing decisions and other words are preserved…')
        word_id=self.item()['bundle_id']
        def worker():
            try:self.jobs.put((True,self.store.append(word_id,seed)))
            except Exception as exc:self.jobs.put((False,str(exc)))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            ok,result=self.jobs.get_nowait();self.busy=False;self.regen.configure(state='normal')
            if ok:self.offset=((len(result['candidates'])-1)//3)*3
            else:messagebox.showerror('Generation failed',result)
            self.refresh()
        except queue.Empty:pass
        self.poll_id=self.root.after(100,self.poll)

def launch(session,store):
    if not session['words']:raise ValueError('Nothing to review')
    # Fail clearly rather than launching a partly prepared, confusing reviewer.
    for item in session['words']:store.load(item['bundle_id'])
    root=tk.Tk();Reviewer(root,session,store);root.mainloop()
