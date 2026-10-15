#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rerun7_glm51.py — 用重洗后修正一审当锚,重跑 7 个一审变动案(真锚×1 + 4扰动)。智谱 GLM-5.1 非思考。
输出 ../results/rerun7_glm51.json。回车贴 key(不留存)。"""
import os, re, json, sys, threading, getpass
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
CASES={c['CLI']:c for c in json.load(open(os.path.join(ROOT,'案件_165案_语义修复版.json')))['cases']}
NEW={r['CLI']:r for r in json.load(open(os.path.join(ROOT,'复核','重洗结果.json')))}
P=json.load(open(os.path.join(ROOT,'prompts.json')))
SEVEN=['CLI.C.565931868','CLI.C.573164263','CLI.C.573703201','CLI.C.573956345','CLI.C.574048559','CLI.C.574750800','CLI.C.575835292']
MODEL='glm-5.1'; BASE='https://open.bigmodel.cn/api/paas/v4'; TAG='glm51'; EXTRA={'thinking':{'type':'disabled'}}; REFFORT=None
OUT=os.path.join(ROOT,'results',f'rerun7_{TAG}.json')
CONDS={'真锚x1':1.0,'真扰动x1/3':1/3.0,'真扰动x2/3':2/3.0,'真扰动x3/2':1.5,'真扰动x3':3.0}
AP={'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉','原审原告、原审被告上诉':'双方上诉'}
def txt(f):
    if isinstance(f,dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')
def base_anchor(cli):
    try: return float(NEW[cli]['一审金额'])
    except: return None
def build(cond,cli):
    c=CASES[cli]; b=base_anchor(cli)
    if b is None or b<0: return None
    f=CONDS[cond]
    if b==0 and cond!='真锚x1': return None      # 零锚只跑真锚×1
    a=round(b*f,2); a=int(a) if a==int(a) else a
    M=P['模板']['带锚二审']
    诉请=txt(c['一审原告诉请']); 查明=txt(c['一审认定事实']); 新增=txt(c['二审新增认定事实'])
    ap=AP.get(c.get('二审上诉方类型'),c.get('二审上诉方类型',''))
    return M['指令'], M['输入_带锚'].format(诉请=诉请,一审查明=查明,二审新增=新增,锚=a,上诉方=ap), a
def parse_num(t):
    t=(t or '').strip().replace(',','').replace('，','')
    m=re.search(r'-?\d+\.?\d*',t)
    if m: return float(m.group())
    if re.search(r'true|驳回|不予支持|不承担|无需赔偿|应当?是?0',t,re.I): return 0.0
    return None
from openai import OpenAI
client=None; lock=threading.Lock()
def call(cond,cli):
    built=build(cond,cli)
    if built is None: return None
    sysmsg,usr,a=built
    kw=dict(model=MODEL,temperature=0,messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}])
    if EXTRA: kw["extra_body"]=EXTRA
    if REFFORT: kw["reasoning_effort"]=REFFORT
    r=client.chat.completions.create(**kw)
    ch=r.choices[0] if getattr(r,'choices',None) else None
    raw=getattr(getattr(ch,'message',None),'content',None)
    if raw is None: raw=f"(空 finish={getattr(ch,'finish_reason','?')})"
    return {"a":cli,"cond":cond,"锚":a,"out":parse_num(raw),"raw":raw,"model":getattr(r,'model',MODEL)}
if __name__=="__main__":
    key=getpass.getpass(f"贴 {TAG} 的 API key(不显示、不留存): ").strip()
    if not key: sys.exit("无 key,退出")
    client=OpenAI(api_key=key,base_url=BASE,timeout=180,max_retries=6)
    tasks=[(cond,cli) for cli in SEVEN for cond in CONDS]
    res=[]
    print(f"{MODEL} 重跑 7 案 × 条件 = 最多 {len(tasks)} 次(零锚案只真锚×1)...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(call,cond,cli):(cond,cli) for cond,cli in tasks}
        for f in as_completed(futs):
            cond,cli=futs[f]
            try:
                rec=f.result()
                if rec: res.append(rec); print(f"  {cond} {cli} 锚{rec['锚']} -> {rec['out']}",flush=True)
            except Exception as e: print(f"  ERR {cond} {cli}: {e!r}",flush=True)
    json.dump(res,open(OUT,'w'),ensure_ascii=False,indent=1)
    print(f"完成 {len(res)} 条 -> {OUT}")
