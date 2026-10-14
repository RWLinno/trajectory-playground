"""Render a shareable GIF from the same unfiltered saved model samples."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment import paths
scene=json.loads((ROOT/'docs/assets/samples.json').read_text())['scenes'][4]
c=scene['condition']
fig,axes=plt.subplots(1,2,figsize=(10,4.7),facecolor='#f4f8fa')
traces=[];dots=[]
for ax,name,color,title in zip(axes,['mlp','diffusion'],['#c66b53','#0d6675'],['MLP · the conditional mean','Diffusion · sampled alternatives']):
    ax.set_facecolor('#ffffff');ax.set_aspect('equal');ax.set_xlim(-1,1);ax.set_ylim(-.8,.8)
    ax.add_patch(plt.Circle((c[1],c[1]*c[2]),c[0],fc='#efba6780',ec='#c59240'))
    p=paths(scene['models'][name]['amplitudes'][:24],c);traces.append(p)
    for path in p:ax.plot(*path.T,c=color,lw=1,alpha=.35)
    ax.scatter([-.9,.9],[0,0],c='#172d39',s=24,zorder=5)
    dots.append(ax.scatter([],[],s=18,c=color,zorder=6))
    ax.set_title(title,loc='left',fontsize=13,color='#172d39',pad=15)
    ax.grid(alpha=.12);ax.tick_params(labelsize=8,colors='#526b79')
    for spine in ax.spines.values():spine.set_color('#d6e2e8')
fig.suptitle('One goal. Two ways around.',fontsize=23,color='#172d39',x=.065,ha='left')
fig.text(.065,.025,'Actual saved samples • synthetic one-coefficient paths • single training seed',fontsize=10,color='#526b79')
fig.tight_layout(rect=[.02,.055,.98,.91])
def animate(frame):
    i=min(64,int(frame/39*64))
    for dot,p in zip(dots,traces):dot.set_offsets(p[:,i])
    return dots
FuncAnimation(fig,animate,frames=40,interval=60).save(ROOT/'docs/assets/preview.gif',writer=PillowWriter(fps=16),dpi=100)
fig.savefig(ROOT/'docs/assets/preview.png',dpi=150)
print('Rendered docs/assets/preview.gif and preview.png')
