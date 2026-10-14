import { trajectory, evaluate } from './geometry.js';
// Exact probability under uniform sampling WITHOUT replacement from this bank.
export function bothRoutes(upper, lower, total, k) {
  const miss = n => { let p=1; for(let i=0;i<k;i++) p *= Math.max(0,n-i)/(total-i); return p; };
  return Math.max(0,Math.min(1,1-miss(total-upper)-miss(total-lower)+miss(total-upper-lower)));
}
export function classify(amplitudes, condition) {
  return amplitudes.map(a=>({a,valid:evaluate([trajectory(a,condition)],condition).valid_rate===1}));
}
const ns='http://www.w3.org/2000/svg';
function el(tag,attrs,text=''){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);e.textContent=text;return e;}
export function diagnostics(scene){
 const plot=document.getElementById('distribution'), curve=document.getElementById('coverage');plot.replaceChildren();curve.replaceChildren();
 const names=['mlp','diffusion','reference'],colors=['#c66b53','#0d6675','#728ba6'];
 const rows=names.map(n=>classify(scene.models[n].amplitudes,scene.condition));
 const bound=Math.max(.7,...rows.flat().map(r=>Math.abs(r.a)))*1.1, x=a=>60+(a/bound+1)*200;
 plot.append(el('line',{x1:x(0),x2:x(0),y1:25,y2:175,stroke:'#b5c6d0','stroke-dasharray':'4 4'}));
 rows.forEach((rs,j)=>{const y=48+j*50;plot.append(el('text',{x:8,y:y-15,fill:colors[j],'font-size':12},names[j]));rs.forEach((r,i)=>plot.append(el('circle',{cx:x(r.a),cy:y+(i%5-2)*3,r:2.8,fill:r.valid?colors[j]:'#bc3e38',opacity:.6})));});
 [-bound,0,bound].forEach(v=>plot.append(el('text',{x:x(v),y:201,'text-anchor':'middle',fill:'#526b79','font-size':11},v.toFixed(2))));
 plot.append(el('text',{x:260,y:222,'text-anchor':'middle',fill:'#526b79','font-size':11},'Bend coefficient a · dots overlap; vertical offset is cosmetic'));
 const X=k=>55+(k-1)/63*390,Y=v=>180-v*145;
 [0,.5,1].forEach(v=>{curve.append(el('line',{x1:55,x2:445,y1:Y(v),y2:Y(v),stroke:'#d6e2e8'}),el('text',{x:45,y:Y(v)+4,'text-anchor':'end',fill:'#526b79','font-size':11},`${v*100}%`));});
 [1,8,16,32,64].forEach(k=>curve.append(el('text',{x:X(k),y:202,'text-anchor':'middle',fill:'#526b79','font-size':11},k)));
 const lines=[];
 rows.forEach((rs,j)=>{const up=rs.filter(r=>r.valid&&r.a>0).length,down=rs.filter(r=>r.valid&&r.a<=0).length;curve.append(el('path',{d:Array.from({length:64},(_,i)=>`${i?'L':'M'}${X(i+1)},${Y(bothRoutes(up,down,rs.length,i+1))}`).join(' '),fill:'none',stroke:colors[j],'stroke-width':2,'stroke-dasharray':j===2?'5 4':''}));lines.push(`${names[j]}: ${up} upper / ${down} lower / ${rs.length-up-down} invalid; P(both, k=8) ${(100*bothRoutes(up,down,rs.length,8)).toFixed(1)}%`);});
 curve.append(el('text',{x:250,y:222,'text-anchor':'middle',fill:'#526b79','font-size':11},'Samples drawn from the 64-path bank (k)'));
 document.getElementById('diagnosticText').textContent=lines.join(' · ');
}
