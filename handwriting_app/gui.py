"""Tkinter stays on the main thread; one background job owns model inference."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from page_engine.common import ROOT
from .service import Settings, Runner, Cancelled, export_pdf, friendly_error

SAMPLE='''# Motion notes

A net force changes the motion of a body. The force equals mass times acceleration.

A greater mass needs more force for the same acceleration. The angle changes over time.

$F = ma$

$a = F/m$

$v = v_0 + at$

$x = 1/2 at^2$

$θ = α + β$

$v^2 = v_0^2 + 2ax$'''

class Application:
    def __init__(self,root,runner=None):
        self.root=root;self.runner=runner or Runner();self.jobs=queue.Queue();self.cancel_event=threading.Event()
        self.busy=False;self.closed=False;self.manifest=None;self.page_index=0;self.dirty=True;self.controls=[]
        root.title('Handwriting AI — local notes to PDF');root.geometry('1320x880');root.minsize(1000,740)
        root.protocol('WM_DELETE_WINDOW',self.close)
        bar=ttk.Frame(root,padding=10);bar.pack(fill='x')
        ttk.Label(bar,text='Handwriting AI',font=('Segoe UI',18,'bold')).pack(side='left')
        ttk.Label(bar,text='Local generation • review spelling • math fallback is labeled').pack(side='left',padx=18)
        self.load_button=ttk.Button(bar,text='Load .txt',command=self.load_text);self.load_button.pack(side='right')
        self.controls.append((self.load_button,'normal'))
        pane=ttk.Panedwindow(root,orient='horizontal');pane.pack(fill='both',expand=True,padx=10)
        editor=ttk.Frame(pane);pane.add(editor,weight=1)
        ttk.Label(editor,text='Notes — blank lines separate paragraphs; # creates a heading').pack(anchor='w')
        self.text=ScrolledText(editor,wrap='word',font=('Segoe UI',11),height=15,undo=True)
        self.text.pack(fill='both',expand=True,pady=6);self.text.insert('1.0',SAMPLE);self.text.edit_modified(False)
        self.text.bind('<<Modified>>',self.text_changed)
        ttk.Label(editor,text='Inline math: $v_0 + at$    Equation paragraph: $a = F/m$\nUse braces for compound scripts: $v^{12}$; denominator groups: $F/(ma)$.\nProse supports English words; place digits/symbols inside $...$.',wraplength=600).pack(anchor='w')
        options=ttk.LabelFrame(editor,text='Page and generation settings',padding=8);options.pack(fill='x',pady=8)
        self.vars={}
        def field(name,title,value,row,col=0,choices=None,width=12):
            variable=tk.StringVar(value=str(value));self.vars[name]=variable
            ttk.Label(options,text=title).grid(row=row,column=col*2,sticky='w',padx=4,pady=3)
            widget=ttk.Combobox(options,textvariable=variable,values=choices,state='readonly',width=width) if choices else ttk.Entry(options,textvariable=variable,width=width)
            widget.grid(row=row,column=col*2+1,sticky='ew',padx=4,pady=3)
            self.controls.append((widget,'readonly' if choices else 'normal'));variable.trace_add('write',lambda *_:self.mark_dirty())
        field('page','Page','A4',0,choices=['A4','Letter']);field('background','Paper','blank',0,1,['blank','lined'])
        field('left','Left margin mm',16,1);field('right','Right margin mm',16,1,1)
        field('top','Top margin mm',22,2);field('bottom','Bottom margin mm',18,2,1)
        field('line_spacing','Line spacing mm',8,3);field('x_height','Writing x-height mm',3.1,3,1)
        field('word_spacing','Word gap mm',2.4,4);field('paragraph_gap','Paragraph gap lines',.8,4,1)
        field('steps','Sampling steps',20,5,choices=[20,30,50]);field('seed','Random seed',9500,5,1)
        field('candidates','Candidates / word',1,6,choices=[1,2,3]);field('device','Device','auto',6,1,['auto','cuda','cpu'])
        models=['Pretrained']
        if (ROOT/'outputs/finetune/erdem_pilot/last_adapter.pt').exists():models.append('Pilot adapter (30 updates)')
        field('model','Model','Pretrained',7,choices=models,width=24)
        self.output=tk.StringVar(value=str(ROOT/'outputs/app_runs'))
        destination=ttk.Frame(editor);destination.pack(fill='x',pady=5)
        ttk.Label(destination,text='Preview folder').pack(side='left')
        entry=ttk.Entry(destination,textvariable=self.output);entry.pack(side='left',fill='x',expand=True,padx=5);self.controls.append((entry,'normal'))
        browse=ttk.Button(destination,text='Browse',command=self.choose_output);browse.pack(side='left');self.controls.append((browse,'normal'))
        actions=ttk.Frame(editor);actions.pack(fill='x',pady=8)
        self.generate_button=ttk.Button(actions,text='Generate preview',command=self.generate);self.generate_button.pack(side='left')
        self.cancel_button=ttk.Button(actions,text='Cancel after current word',command=self.cancel,state='disabled');self.cancel_button.pack(side='left',padx=6)
        self.export_button=ttk.Button(actions,text='Export PDF…',command=self.export,state='disabled');self.export_button.pack(side='left')
        preview=ttk.Frame(pane);pane.add(preview,weight=1)
        nav=ttk.Frame(preview);nav.pack(fill='x')
        ttk.Button(nav,text='◀ Previous',command=lambda:self.navigate(-1)).pack(side='left')
        self.page_label=ttk.Label(nav,text='No preview yet');self.page_label.pack(side='left',padx=12)
        ttk.Button(nav,text='Next ▶',command=lambda:self.navigate(1)).pack(side='left')
        self.zoom=tk.StringVar(value='Fit')
        zoom=ttk.Combobox(nav,textvariable=self.zoom,values=['Fit','75%','100%','150%'],state='readonly',width=6);zoom.pack(side='right')
        zoom.bind('<<ComboboxSelected>>',lambda _:self.show_page())
        view=ttk.Frame(preview);view.pack(fill='both',expand=True,pady=8)
        view.rowconfigure(0,weight=1);view.columnconfigure(0,weight=1)
        self.canvas=tk.Canvas(view,bg='#e5e8eb',highlightthickness=0);self.canvas.grid(row=0,column=0,sticky='nsew')
        sy=ttk.Scrollbar(view,orient='vertical',command=self.canvas.yview);sy.grid(row=0,column=1,sticky='ns')
        sx=ttk.Scrollbar(view,orient='horizontal',command=self.canvas.xview);sx.grid(row=1,column=0,sticky='ew')
        self.canvas.configure(xscrollcommand=sx.set,yscrollcommand=sy.set)
        self.canvas.bind('<Configure>',lambda _:self.show_page())
        self.provenance=ttk.Label(preview,text='Approved math symbols are optional. Missing symbols use standard fonts.',wraplength=560)
        self.provenance.pack(fill='x')
        self.progress=ttk.Progressbar(root,maximum=100);self.progress.pack(fill='x',padx=10,pady=6)
        self.status=tk.StringVar(value='Ready. Choose settings, generate, inspect all pages, then export.')
        ttk.Label(root,textvariable=self.status,wraplength=1250).pack(fill='x',padx=10,pady=(0,10))
        self.poll_id=root.after(100,self.poll)

    def mark_dirty(self):
        self.dirty=True
        if hasattr(self,'export_button'):self.export_button.configure(state='disabled')
        if self.manifest:self.status.set('Notes/settings changed. Regenerate to update the preview and enable export.')

    def text_changed(self,event=None):
        if self.text.edit_modified():self.mark_dirty();self.text.edit_modified(False)

    def settings(self):
        values={key:variable.get() for key,variable in self.vars.items()}
        adapter='outputs/finetune/erdem_pilot/last_adapter.pt' if values['model']!='Pretrained' else None
        try:
            settings=Settings(page=values['page'],background=values['background'],margins=tuple(float(values[k]) for k in ('left','right','top','bottom')),
                line_spacing=float(values['line_spacing']),x_height=float(values['x_height']),word_spacing=float(values['word_spacing']),
                paragraph_gap=float(values['paragraph_gap']),steps=int(values['steps']),seed=int(values['seed']),
                candidates=int(values['candidates']),device=values['device'],adapter=adapter)
            settings.validate();return settings
        except (TypeError,ValueError) as error:raise ValueError('Check page settings: '+str(error)) from error

    def set_busy(self,busy,can_cancel=False):
        self.busy=busy;self.generate_button.configure(state='disabled' if busy else 'normal')
        self.export_button.configure(state='normal' if not busy and self.manifest and not self.dirty else 'disabled')
        self.cancel_button.configure(state='normal' if busy and can_cancel else 'disabled')
        self.text.configure(state='disabled' if busy else 'normal')
        for widget,state in self.controls:widget.configure(state='disabled' if busy else state)

    def generate(self):
        if self.busy:return
        try:
            settings=self.settings();text=self.text.get('1.0','end-1c');output=self.output.get()
            if not text.strip():raise ValueError('Enter notes before generating')
        except Exception as error:messagebox.showerror('Check input',str(error));return
        self.cancel_event.clear();self.set_busy(True,True);self.progress['value']=0;self.status.set('Starting background generation…')
        def worker():
            try:
                result=self.runner.generate(text,settings,output,lambda *a:self.jobs.put(('progress',a)),self.cancel_event)
                self.jobs.put(('generated',result))
            except Cancelled as error:self.jobs.put(('cancelled',str(error)))
            except Exception as error:self.jobs.put(('error',friendly_error(error)))
        threading.Thread(target=worker,daemon=True).start()

    def cancel(self):
        self.cancel_event.set();self.cancel_button.configure(state='disabled');self.status.set('Cancellation requested; the current inference call will finish first.')

    def export(self):
        if self.busy or not self.manifest or self.dirty:return
        path=filedialog.asksaveasfilename(defaultextension='.pdf',filetypes=[('PDF document','*.pdf')],initialdir=self.output.get(),initialfile='handwritten_notes.pdf')
        if not path:return
        self.set_busy(True);self.status.set('Exporting the exact preview pages…')
        manifest=self.manifest
        def worker():
            try:self.jobs.put(('exported',str(export_pdf(manifest,path))))
            except Exception as error:self.jobs.put(('error',friendly_error(error)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        if self.closed:return
        try:
            while True:
                event,value=self.jobs.get_nowait()
                if event=='progress':
                    done,total,message=value;self.progress['value']=done/total*100
                    if not self.cancel_event.is_set():self.status.set(message)
                elif event=='generated':
                    self.manifest=value;self.page_index=0;self.dirty=False;self.set_busy(False);self.show_page()
                    self.progress['value']=100
                    self.status.set(f'Ready: {len(value["pages"])} pages; {value["cache_hits"]} cache hits, {value["new_words"]} new candidates. Review spelling. Run saved in {value["app"]["run_directory"]}')
                elif event=='exported':self.set_busy(False);self.status.set('PDF saved: '+value)
                elif event=='cancelled':self.set_busy(False);self.status.set(value)
                else:self.set_busy(False);self.status.set('Task failed; adjust settings/files and retry.');messagebox.showerror('Recoverable error',value)
        except queue.Empty:pass
        self.poll_id=self.root.after(100,self.poll)

    def show_page(self):
        if not self.manifest:return
        path=ROOT/self.manifest['pages'][self.page_index]
        try:
            with Image.open(path) as raw:
                image=raw.convert('RGB')
                if self.zoom.get()=='Fit':image.thumbnail((max(50,self.canvas.winfo_width()-24),max(50,self.canvas.winfo_height()-24)),Image.Resampling.LANCZOS)
                else:
                    factor=float(self.zoom.get().rstrip('%'))/100
                    image=image.resize((round(image.width*factor),round(image.height*factor)),Image.Resampling.LANCZOS)
            self.photo=ImageTk.PhotoImage(image);self.canvas.delete('all')
            if self.zoom.get()=='Fit':
                self.canvas.configure(scrollregion=(0,0,self.canvas.winfo_width(),self.canvas.winfo_height()))
                self.canvas.xview_moveto(0);self.canvas.yview_moveto(0)
                self.canvas.create_image(self.canvas.winfo_width()//2,self.canvas.winfo_height()//2,image=self.photo,anchor='center')
            else:
                self.canvas.configure(scrollregion=(0,0,image.width+24,image.height+24))
                self.canvas.create_image(12,12,image=self.photo,anchor='nw')
            self.page_label.configure(text=f'Page {self.page_index+1} / {len(self.manifest["pages"])}')
            equations=[p for p in self.manifest['positions'] if p['page']==self.page_index+1 and p['rendering']['source']!='diffusionpen_word']
            missing=sorted({s for p in equations for s in p['rendering'].get('missing_symbols',[])})
            fallback=any(p['rendering']['source']=='typeset_math_not_handwriting' for p in equations)
            message=('Standard-font math fallback (not handwriting). Missing approved symbols: '+', '.join(missing)) if fallback else 'Math uses approved symbol crops where present. Prose spelling still needs visual review.'
            flagged=sorted({p['text'] for p in self.manifest['positions'] if p['page']==self.page_index+1 and p['rendering'].get('flags')})
            if flagged:message+=' Structurally flagged words: '+', '.join(flagged)+'. See the run manifest/reviewer.'
            self.provenance.configure(text=message)
        except Exception as error:self.status.set('Cannot display page: '+str(error))

    def navigate(self,delta):
        if self.manifest:self.page_index=max(0,min(len(self.manifest['pages'])-1,self.page_index+delta));self.show_page()

    def load_text(self):
        path=filedialog.askopenfilename(filetypes=[('Text notes','*.txt')])
        if not path:return
        try:
            if Path(path).stat().st_size>500_000:raise ValueError('Text file too large; split it into smaller documents')
            text=Path(path).read_text(encoding='utf-8-sig');self.text.delete('1.0','end');self.text.insert('1.0',text);self.mark_dirty()
        except Exception as error:messagebox.showerror('Cannot load notes',str(error)+'\nSave the file as UTF-8 text.')

    def choose_output(self):
        path=filedialog.askdirectory(initialdir=self.output.get())
        if path:self.output.set(path)

    def close(self):
        if self.busy:
            self.cancel();self.status.set('Wait for the current job to finish/cancel before closing; completed cache entries will be retained.');return
        self.closed=True;self.root.after_cancel(self.poll_id);self.root.destroy()

def launch():
    root=tk.Tk();Application(root);root.mainloop()
