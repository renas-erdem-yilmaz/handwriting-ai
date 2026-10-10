"""GUI launch, with a headless path using the same service for reproducibility."""
import argparse
import sys
from pathlib import Path
from .service import Runner, Settings, export_pdf, friendly_error

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(errors='backslashreplace')
    parser=argparse.ArgumentParser(description='Local Handwriting AI desktop application')
    parser.add_argument('--text-file');parser.add_argument('--output',default='outputs/app_runs')
    parser.add_argument('--pdf');parser.add_argument('--page',choices=['A4','Letter'],default='A4')
    parser.add_argument('--paper',choices=['blank','lined'],default='blank');parser.add_argument('--seed',type=int,default=9500)
    parser.add_argument('--adapter');parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if not args.text_file:
        if args.pdf:parser.error('--pdf requires --text-file')
        from .gui import launch
        launch();return
    try:
        text=Path(args.text_file).read_text(encoding='utf-8-sig')
        settings=Settings(page=args.page,background=args.paper,seed=args.seed,adapter=args.adapter,device=args.device)
        result=Runner().generate(text,settings,args.output,lambda n,total,message:print(message,flush=True))
        print('Preview:',result['pages'])
        if args.pdf:print('PDF:',export_pdf(result,args.pdf))
    except Exception as error:parser.exit(1,friendly_error(error)+'\n')

if __name__=='__main__':main()
