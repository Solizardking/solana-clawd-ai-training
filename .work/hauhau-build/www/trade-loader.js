(function(){
  let promise,ready=false,pendingToken=null;
  const target=document.getElementById('trade');
  if(!target)return;
  function load(){
    if(!promise)promise=import('/trade.js').then(function(){ready=true;if(pendingToken){const detail=pendingToken;pendingToken=null;document.dispatchEvent(new CustomEvent('hauhau:trade-token',{detail:detail}));}}).catch(function(){promise=null;const el=document.getElementById('ht-message');if(el)el.textContent='Trading could not load. Reload the page to retry.';});
    return promise;
  }
  if('IntersectionObserver' in window){const observer=new IntersectionObserver(function(entries){if(entries.some(function(e){return e.isIntersecting;})){observer.disconnect();load();}},{rootMargin:'350px'});observer.observe(target);}else load();
  target.addEventListener('pointerdown',load,{once:true});
  document.addEventListener('hauhau:trade-token',function(event){if(!ready){pendingToken=event.detail;event.stopImmediatePropagation();load();}});
})();
