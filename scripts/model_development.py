"""End-to-end Python model-development implementation.

Implements LASSO, binary Flamingo-style search, binary Bat Algorithm,
hybrid Flamingo-Bat selection, and RF/SVM/XGBoost evaluation.

NOTE: model.zip supplied outputs but not the original lost source code/raw matrix.
This is a transparent reproducible implementation based on the manuscript, not a
claim to be the exact historical code that generated the supplied outputs.
"""
from __future__ import annotations
import argparse, json, math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

@dataclass
class Config:
    seed:int=2026; outer_folds:int=5; inner_folds:int=3
    prefilter:int|None=500; population_size:int=12; iterations:int=30
    accuracy_weight:float=.90; subset_weight:float=.10; min_features:int=2; n_jobs:int=-1

def seed_everything(seed): random.seed(seed); np.random.seed(seed)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-60,60)))
def repair(bits, minimum, rng):
    bits=(np.asarray(bits)>.5).astype(np.int8)
    if bits.sum()<minimum:
        z=np.flatnonzero(bits==0); k=min(minimum-int(bits.sum()),len(z))
        if k: bits[rng.choice(z,k,replace=False)]=1
    return bits

def load_expression(path,label='label'):
    df=pd.read_csv(path)
    if label not in df: raise ValueError(f"Missing '{label}' column; expected rows=samples, columns=genes.")
    y=df.pop(label); X=df.apply(pd.to_numeric,errors='coerce'); X=X.loc[:,X.notna().any()]
    if not pd.api.types.is_numeric_dtype(y):
        cls=list(pd.unique(y.dropna()));
        if len(cls)!=2: raise ValueError('Exactly two classes are required.')
        y=y.map({cls[0]:0,cls[1]:1})
    y=pd.to_numeric(y).astype(int)
    if y.nunique()!=2: raise ValueError('Exactly two classes are required.')
    return X,y,list(X.columns)

def classifier_spaces(cfg):
    m={'Random_forest':(RandomForestClassifier(random_state=cfg.seed,n_jobs=cfg.n_jobs,class_weight='balanced'),
       {'model__n_estimators':[200],'model__max_depth':[None,3,5],'model__min_samples_leaf':[1,2]}),
       'SVM':(SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=cfg.seed),
       {'model__C':[.1,1,10],'model__gamma':['scale','auto']})}
    if XGBClassifier:
        m['XGBoost']=(XGBClassifier(objective='binary:logistic',eval_metric='logloss',random_state=cfg.seed,n_jobs=cfg.n_jobs),
          {'model__n_estimators':[100,200],'model__max_depth':[2,3],'model__learning_rate':[.05,.1],
           'model__subsample':[.8,1.0],'model__colsample_bytree':[.8,1.0]})
    return m

def best_model(X,y,name,cfg):
    model,grid=classifier_spaces(cfg); cv=StratifiedKFold(cfg.inner_folds,shuffle=True,random_state=cfg.seed)
    s=GridSearchCV(Pipeline([('model',model)]),grid,scoring='f1',cv=cv,n_jobs=cfg.n_jobs)
    s.fit(X,y); return s.best_estimator_,s.best_score_

def metric_dict(y,p): return {'Accuracy':accuracy_score(y,p),'Precision':precision_score(y,p,zero_division=0),'Recall':recall_score(y,p,zero_division=0),'F1 Score':f1_score(y,p,zero_division=0),'F2 Score':fbeta_score(y,p,beta=2,zero_division=0)}

class Objective:
    def __init__(self,X,y,cfg,seed): self.X=X; self.y=np.asarray(y); self.cfg=cfg; self.rng=np.random.default_rng(seed); self.cache={}
    def __call__(self,bits):
        bits=repair(bits,self.cfg.min_features,self.rng); key=tuple(bits.tolist())
        if key in self.cache:return self.cache[key]
        idx=np.flatnonzero(bits); Xs=self.X[:,idx]; cv=StratifiedKFold(self.cfg.inner_folds,shuffle=True,random_state=self.cfg.seed); scores=[]
        for tr,va in cv.split(Xs,self.y):
            m=SVC(C=1,kernel='rbf',gamma='scale',class_weight='balanced'); m.fit(Xs[tr],self.y[tr]); scores.append(f1_score(self.y[va],m.predict(Xs[va]),zero_division=0))
        f1=float(np.mean(scores)); frac=len(idx)/self.X.shape[1]; fit=self.cfg.accuracy_weight*f1+self.cfg.subset_weight*(1-frac); self.cache[key]=fit; return fit

def lasso_select(X,y,cfg):
    pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(penalty='l1',solver='liblinear',max_iter=5000,random_state=cfg.seed))])
    cv=StratifiedKFold(cfg.inner_folds,shuffle=True,random_state=cfg.seed)
    s=GridSearchCV(pipe,{'model__C':np.logspace(-3,2,12)},scoring='f1',cv=cv,n_jobs=cfg.n_jobs); s.fit(X,y)
    c=np.abs(s.best_estimator_.named_steps['model'].coef_[0]); idx=np.flatnonzero(c>1e-10)
    if len(idx)<cfg.min_features: idx=np.argsort(c)[-cfg.min_features:]
    return idx,s.best_score_

def flamingo(X,y,cfg,seed):
    rng=np.random.default_rng(seed); d=X.shape[1]; n=cfg.population_size; obj=Objective(X,y,cfg,seed); pop=(rng.random((n,d))>.5).astype(float); best=None; bf=-np.inf
    for t in range(cfg.iterations):
        ex=1-t/max(1,cfg.iterations-1)
        for p in pop:
            b=repair(p,cfg.min_features,rng); f=obj(b)
            if f>bf:bf,bf2=f,f;best=b.copy()
        new=pop.copy()
        for i in range(n):
            peer=pop[rng.integers(n)]; direction=best-pop[i]; move=ex*rng.normal(size=d)*(peer-pop[i])+(1-ex)*rng.uniform(-1,1,d)*direction
            new[i]=(rng.random(d)<sigmoid(pop[i]+move-.5)).astype(float)
        pop=new
    return np.flatnonzero(best),bf

def bat(X,y,cfg,seed):
    rng=np.random.default_rng(seed); d=X.shape[1]; n=cfg.population_size; obj=Objective(X,y,cfg,seed); pos=(rng.random((n,d))>.5).astype(float); vel=np.zeros((n,d)); loud=np.full(n,.9); pulse=np.full(n,.5); best=None;bf=-np.inf
    for t in range(cfg.iterations):
        for i in range(n):
            b=repair(pos[i],cfg.min_features,rng);f=obj(b)
            if f>bf:bf=f;best=b.copy()
        for i in range(n):
            freq=2*rng.random();vel[i]+=(pos[i]-best)*freq; cand=pos[i]+vel[i]
            if rng.random()>pulse[i]:cand=best+.01*rng.normal(size=d)
            b=repair(rng.random(d)<sigmoid(cand),cfg.min_features,rng);f=obj(b)
            if f>=obj(repair(pos[i],cfg.min_features,rng)) and rng.random()<loud[i]:pos[i]=b;loud[i]*=.95;pulse[i]=.5*(1-math.exp(-.05*(t+1)))
    return np.flatnonzero(best),bf

def hybrid(X,y,cfg,seed):
    rng=np.random.default_rng(seed); d=X.shape[1];n=cfg.population_size;obj=Objective(X,y,cfg,seed);pos=(rng.random((n,d))>.5).astype(float);vel=np.zeros((n,d));loud=np.full(n,.9);pulse=np.full(n,.5);best=None;bf=-np.inf
    for t in range(cfg.iterations):
        ex=1-t/max(1,cfg.iterations-1)
        for p in pos:
            b=repair(p,cfg.min_features,rng);f=obj(b)
            if f>bf:bf=f;best=b.copy()
        # Flamingo global exploration
        for i in range(n):
            peer=pos[rng.integers(n)];move=ex*rng.normal(size=d)*(peer-pos[i])+(1-ex)*rng.uniform(-1,1,d)*(best-pos[i]);pos[i]=(rng.random(d)<sigmoid(pos[i]+move-.5)).astype(float)
        # Bat local exploitation
        for i in range(n):
            cur=repair(pos[i],cfg.min_features,rng);cf=obj(cur);vel[i]+=(pos[i]-best)*(2*rng.random());cand=pos[i]+vel[i]
            if rng.random()>pulse[i]:cand=best+.01*rng.normal(size=d)
            b=repair(rng.random(d)<sigmoid(cand),cfg.min_features,rng);f=obj(b)
            if f>=cf and rng.random()<loud[i]:pos[i]=b;loud[i]*=.95;pulse[i]=.5*(1-math.exp(-.05*(t+1)))
            if f>bf:bf=f;best=b.copy()
        pos[0]=best
    return np.flatnonzero(best),bf

def evaluate(X,y,idx,method,cfg):
    Xs=X[:,idx];outer=StratifiedKFold(cfg.outer_folds,shuffle=True,random_state=cfg.seed);rows=[]
    for fold,(tr,te) in enumerate(outer.split(Xs,y),1):
        for name in classifier_spaces(cfg):
            m,_=best_model(Xs[tr],y.iloc[tr] if hasattr(y,'iloc') else y[tr],name,cfg);p=m.predict(Xs[te]); rows.append({'fold':fold,'feature_selection':method,'classifier':name,'n_genes':len(idx),**metric_dict(y.iloc[te] if hasattr(y,'iloc') else y[te],p)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expression',required=True);ap.add_argument('--label-column',default='label');ap.add_argument('--output',default='results/reproduced_model');ap.add_argument('--prefilter',type=int,default=500);ap.add_argument('--population',type=int,default=12);ap.add_argument('--iterations',type=int,default=30);ap.add_argument('--seed',type=int,default=2026);a=ap.parse_args()
    cfg=Config(seed=a.seed,prefilter=None if a.prefilter<=0 else a.prefilter,population_size=a.population,iterations=a.iterations);seed_everything(cfg.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    Xdf,y,genes=load_expression(a.expression,a.label_column);X=SimpleImputer(strategy='median').fit_transform(Xdf);X=StandardScaler().fit_transform(X)
    if cfg.prefilter and cfg.prefilter<X.shape[1]:
        s=SelectKBest(f_classif,k=cfg.prefilter);X=s.fit_transform(X,y);genes=list(np.asarray(genes)[s.get_support()])
    methods={'LASSO':lasso_select(X,y,cfg)[0],'Flamingo':flamingo(X,y,cfg,cfg.seed+11)[0],'Bat':bat(X,y,cfg,cfg.seed+22)[0],'Hybrid Flamingo-Bat':hybrid(X,y,cfg,cfg.seed+33)[0]}
    allrows=[]
    for method,idx in methods.items():
        pd.DataFrame({'Gene':[genes[i] for i in idx]}).to_csv(out/(method.lower().replace(' ','_').replace('-','_')+'_selected_genes.csv'),index=False);allrows.append(evaluate(X,y,idx,method,cfg))
    perf=pd.concat(allrows,ignore_index=True);perf.to_csv(out/'model_performance_by_fold.csv',index=False);perf.groupby(['feature_selection','classifier','n_genes'])[['Accuracy','Precision','Recall','F1 Score','F2 Score']].mean().reset_index().to_csv(out/'model_performance_summary.csv',index=False)
    (out/'configuration.json').write_text(json.dumps(cfg.__dict__,indent=2));print('Completed.');print(perf.groupby(['feature_selection','classifier','n_genes'])[['Accuracy','Precision','Recall','F1 Score','F2 Score']].mean())
if __name__=='__main__':main()
