#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rerun1z_dsv4pro.py — 补跑 rerun7 漏掉的第 7 案 CLI.C.575835292:修正后一审额为 0,原跑显示了旧值 200。只跑真锚×1(锚=0),1 次。
DeepSeek-v4-pro 非思考,参数与 7/1 的 run_all.py 主跑一致。零一审额案不做乘法扰动(不在 149 框架内)。
输出 ../results/rerun1z_dsv4pro.json。回车贴 key(不显示、不留存)。"""
import os, re, json, sys, threading, getpass, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
CASES={c['CLI']:c for c in json.load(open(os.path.join(ROOT,'案件_165案_语义修复版.json')))['cases']}
NEW={r['CLI']:r for r in json.load(open(os.path.join(ROOT,'复核','重洗结果.json')))}
P=json.load(open(os.path.join(ROOT,'prompts.json')))
SIX=['CLI.C.575835292']
MODEL='deepseek-v4-pro'; BASE=os.environ.get('RUN_BASE','https://api.deepseek.com'); TAG='dsv4pro'
EXTRA={'thinking':{'type':'disabled'}}
OUT=os.path.join(ROOT,'results',f'rerun1z_{TAG}.json')
CONDS={'真锚x1':1.0}
AP={'原审原告上诉':'原告上诉','原审被告上诉':'被告上诉','双方上诉':'双方上诉','原审原告、原审被告上诉':'双方上诉'}
def txt(f):
    if isinstance(f,dict):
        for k in ['脱敏','原文','text']:
            if f.get(k): return str(f[k])
        return ''
    return str(f or '')
def build(cond,cli):
    c=CASES[cli]; b=float(NEW[cli]['一审金额'])
    a=round(b*CONDS[cond],2); a=int(a) if a==int(a) else a
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
client=None
def call(cond,cli):
    sysmsg,usr,a=build(cond,cli)
    r=client.chat.completions.create(model=MODEL,temperature=0,
        messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}],extra_body=EXTRA)
    ch=r.choices[0] if getattr(r,'choices',None) else None
    raw=getattr(getattr(ch,'message',None),'content',None)
    if raw is None: raw=f"(空 finish={getattr(ch,'finish_reason','?')})"
    return {"a":cli,"cond":cond,"锚":a,"out":parse_num(raw),"raw":raw,"model":getattr(r,'model',MODEL),
            "run_at":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
if __name__=="__main__":
    key=getpass.getpass(f"贴 {TAG} 的 API key(不显示、不留存): ").strip()
    if not key: sys.exit("无 key,退出")
    client=OpenAI(api_key=key,base_url=BASE,timeout=180,max_retries=6)
    tasks=[(cond,cli) for cli in SIX for cond in CONDS]
    print(f"{MODEL} @ {BASE} 补跑 1 案 × 真锚 = {len(tasks)} 次 ...")
    res=[]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(call,cond,cli):(cond,cli) for cond,cli in tasks}
        for f in as_completed(futs):
            cond,cli=futs[f]
            try:
                rec=f.result(); res.append(rec)
                print(f"  {cond} {cli} 锚{rec['锚']} -> {rec['out']}",flush=True)
            except Exception as e: print(f"  ERR {cond} {cli}: {e!r}",flush=True)
    json.dump(res,open(OUT,'w'),ensure_ascii=False,indent=1)
    print(f"完成 {len(res)}/{len(tasks)} 条 -> {OUT}")
