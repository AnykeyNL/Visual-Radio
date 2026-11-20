const {chromium}=require('playwright');
(async()=>{const [,,scene,w,h,t,out,arg]=process.argv;const b=await chromium.launch();const p=await b.newPage({viewport:{width:+w,height:+h}});
await p.goto(require('url').pathToFileURL(require('path').join(__dirname,'gfx.html')).href);await p.evaluate(()=>document.fonts.ready);await p.evaluate(()=>document.body.classList.add('solid'));
for(const tt of t.split(',')){await p.evaluate(([s,x,a])=>render(s,x,a),[scene,+tt,arg||'']);}
await p.screenshot({path:out});await b.close()})();
