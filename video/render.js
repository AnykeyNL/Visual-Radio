// node render.js scene W H seconds outdir [arg]  -> PNG frames at 30fps (or single PNG if seconds==0)
const {chromium}=require('playwright');
(async()=>{const [,,scene,w,h,secs,out,arg]=process.argv;const fs=require('fs');fs.mkdirSync(out,{recursive:true});
const b=await chromium.launch();const p=await b.newPage({viewport:{width:+w,height:+h},deviceScaleFactor:1});
await p.goto(require('url').pathToFileURL(require('path').join(__dirname,'gfx.html')).href);await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(300);
const solid=+secs>0;if(solid)await p.evaluate(()=>document.body.classList.add('solid'));
const n=solid?Math.round(+secs*30):1;
for(let i=0;i<n;i++){await p.evaluate(([s,t,a])=>render(s,t,a),[scene,i/30,arg||'']);await p.screenshot({path:`${out}/${solid?String(i).padStart(4,'0'):'frame'}.png`,omitBackground:!solid})}
await b.close()})();
