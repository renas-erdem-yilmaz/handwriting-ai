"""Measure foreground and lowercase body scale without modifying glyph geometry."""
import re
import cv2
import numpy as np
from PIL import Image

def white_image(image):
    rgba=image.convert('RGBA')
    background=Image.new('RGBA',rgba.size,'white');background.alpha_composite(rgba)
    return background.convert('L')

def category(text):
    letters=re.sub('[^A-Za-z]','',text)
    if letters.isupper():return 'capitals'
    asc=any(c in 'bdfhklt' or c.isupper() for c in letters)
    desc=any(c in 'gjpqy' for c in letters)
    return 'both' if asc and desc else 'ascenders' if asc else 'descenders' if desc else 'body'

PRIORS={'body':1.,'ascenders':.67,'descenders':.72,'both':.55,'capitals':.70}

def measure(image,text):
    image=white_image(image);a=np.asarray(image)
    background=float(np.percentile(a,98))
    dark=a[a<190]
    if not len(dark):raise ValueError('No visible handwriting ink')
    threshold=min(210,background-max(35,(background-float(np.percentile(dark,30)))*.30))
    mask=(a<threshold).astype('uint8')
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max())
    significant=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=max(4,largest*.012)]
    if not significant:raise ValueError('No significant ink components')
    ys,xs=np.where(np.isin(labels,significant));x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
    h=y1-y0
    # Preserve neighboring punctuation and i/j dots; ignore remote isolated speckles.
    keep=list(significant)
    for i in range(1,n):
        x,y,w,hh,area=stats[i]
        if area>=2 and x+w>=x0-.3*h and x<=x1+.5*h and y+hh>=y0-.45*h and y<=y1+.2*h:
            keep.append(i)
    clean=np.isin(labels,keep)
    ys,xs=np.where(clean);box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
    core=clean[box[1]:box[3],box[0]:box[2]]
    height,width=core.shape
    mass=core.sum(axis=0);cols=np.flatnonzero(mass>=max(2,height*.08))
    if text[-1:] in '.,;:!?':cols=cols[cols<width*.90]
    kind=category(text)
    baseline=height-1;xh=height*PRIORS[kind];confidence=False;reason='lexical-extent-prior'
    if len(cols)>=5:
        tops=np.array([np.flatnonzero(core[:,c])[0] for c in cols])
        bottoms=np.array([np.flatnonzero(core[:,c])[-1] for c in cols])
        def mode(values):
            hist=np.bincount(values,minlength=height).astype(float)
            radius=max(1,round(height*.045))
            smooth=np.convolve(hist,np.ones(radius*2+1),mode='same')
            peak=int(np.argmax(smooth))
            near=values[abs(values-peak)<=radius]
            return int(round(np.median(near))),float(len(near)/len(values))
        bottom,support_bottom=mode(bottoms)
        top,support_top=mode(tops)
        baseline=int(np.clip(bottom,round(height*.45),height-1))
        observed=baseline-top+1
        ordinary=sum(c in 'acemnorsuvwxz' for c in re.sub('[^a-z]','',text))
        fraction=observed/height
        enough_body=(ordinary>=2 or (ordinary>=1 and support_top>=.40 and support_bottom>=.40))
        confidence=(enough_body and support_top>=.22 and support_bottom>=.22 and .46<=fraction<=1.0 and observed>=5 and kind!='capitals')
        if confidence:xh=float(observed);reason='lowercase-top-and-baseline-modes'
        elif kind in ('descenders','both') and bottom>=height*.92:
            baseline=round(height*(.78 if kind=='both' else .73))-1
            reason='descender-extent-fallback'
    return {'box':list(box),'mask':core,'baseline':baseline,'x_height':float(xh),'confidence':confidence,
            'category':kind,'method':reason,'ink_height':height,'ink_width':width}

def calibrate(measurements):
    ratios=dict(PRIORS);counts={k:0 for k in PRIORS}
    for kind in ratios:
        values=[m['x_height']/m['ink_height'] for m in measurements if m['confidence'] and m['category']==kind]
        counts[kind]=len(values)
        if len(values)>=3:
            ratios[kind]=float(np.clip(np.median(values),max(.46,PRIORS[kind]-.15),min(1.,PRIORS[kind]+.15)))
    return {'fractions':ratios,'reliable_word_counts':counts}
