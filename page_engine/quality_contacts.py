"""Raw candidate contact sheets for explicit visual comparison, no inference."""
from PIL import Image,ImageDraw,ImageFont
from .common import ROOT,read_json

def main():
    session=read_json(ROOT/'outputs/quality_tests/review_session.json')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for group in range((len(session['words'])+5)//6):
        rows=session['words'][group*6:(group+1)*6]
        canvas=Image.new('RGB',(1140,len(rows)*140+35),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,5),'Raw DiffusionPen candidates — requested text is a label, NOT proof of correct generation',font=font,fill='black')
        for row,item in enumerate(rows):
            bundle=read_json(ROOT/'.reforge-cache/candidate_reviews'/f'{item["bundle_id"]}.json')
            for column,candidate in enumerate(bundle['candidates'][:3]):
                x=column*380+10;y=row*140+38
                draw.text((x,y),f'#{item["index"]} {item["text"]} | candidate {column+1}',font=font,fill='black')
                image=Image.open(ROOT/candidate['image']).convert('RGB')
                scale=min(350/image.width,80/image.height)
                image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
                canvas.paste(image,(x,y+28))
                draw.text((x,y+107),f'Seed {candidate["seed"]}',font=font,fill='black')
        path=ROOT/f'outputs/quality_tests/candidates_{group+1:02d}.png'
        if path.exists():raise FileExistsError(path)
        canvas.save(path)

if __name__=='__main__':main()
