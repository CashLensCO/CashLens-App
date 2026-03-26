"""inference_coins.py — python inference_coins.py imagen.jpg"""
import sys, cv2, numpy as np
from pathlib import Path
IMG_SIZE, CLASSES = 224, ["50","100","200","500","1000"]
def norm(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(3.0,(8,8)).apply(l); return cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)
def crop(img,pad=.2):
    h,w=img.shape[:2]; sc=1.0
    if max(h,w)>640: sc=640/max(h,w); sm=cv2.resize(img,(int(w*sc),int(h*sc)))
    else: sm=img
    g=cv2.GaussianBlur(cv2.cvtColor(norm(sm),cv2.COLOR_BGR2GRAY),(5,5),0); sh,sw=sm.shape[:2]; md=min(sh,sw)
    for p2 in [30,20,40]:
        c=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,md//3,param1=80,param2=p2,minRadius=int(md*.04),maxRadius=int(md*.55))
        if c is not None:
            b=max(c[0],key=lambda x:x[2]); cx,cy,r=int(b[0]/sc),int(b[1]/sc),int(b[2]/sc); rp=int(r*(1+pad))
            cr=img[max(0,cy-rp):min(h,cy+rp),max(0,cx-rp):min(w,cx+rp)]
            if cr.shape[0]>10 and cr.shape[1]>10: return cr
    s=min(h,w); return img[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]
def tta(interp,img,n=4):
    i,o=interp.get_input_details(),interp.get_output_details(); ps=[]
    for k in range(n):
        a=img.copy()
        if k>0: M=cv2.getRotationMatrix2D((a.shape[1]//2,a.shape[0]//2),k*90,1); a=cv2.warpAffine(a,M,(a.shape[1],a.shape[0]))
        c=crop(a); x=cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),(IMG_SIZE,IMG_SIZE)).astype(np.float32)
        interp.set_tensor(i[0]["index"],np.expand_dims(x,0)); interp.invoke(); ps.append(interp.get_tensor(o[0]["index"])[0])
    return np.mean(ps,axis=0)
if __name__=="__main__":
    import tensorflow as tf; it=tf.lite.Interpreter("models/coins_v2.tflite"); it.allocate_tensors()
    for p in sys.argv[1:]:
        p=Path(p); fs=list(p.glob("*.*")) if p.is_dir() else [p]
        for f in fs:
            img=cv2.imread(str(f))
            if img is None: continue
            pr=tta(it,img); ix=np.argmax(pr); print(f"{f.name}: ${CLASSES[ix]} ({pr[ix]*100:.1f}%)")
