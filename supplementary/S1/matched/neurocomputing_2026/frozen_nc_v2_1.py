#!/usr/bin/env python3
"""NC-ABL-v2.1: three-arm matched-setting TCR confirmation.

Operations (run from any directory; does not modify the manuscript repo):

  python nc_three_arm_confirm_v2_1.py --stage smoke
  python nc_three_arm_confirm_v2_1.py --stage calibrate
  python nc_three_arm_confirm_v2_1.py --stage pilot --calibration results_calibrate/calibration.json
  python nc_three_arm_confirm_v2_1.py --stage confirm --calibration results_calibrate/calibration.json

Only the primary A/B/C matched-hyperparameter arms are evaluated. The
reference calibration from NC-ABL-v2 is reused without changes. Confirmation
requires --confirm-frozen. Scientific protocol: NC_ABLATION_PROTOCOL_v2_1.md.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TAU_MIN, TAU_MAX = 0.08, 3.50
FULL_TASKS = ("memory_d10", "narma10", "lorenz_x")
ARMS = ("row_rms", "row_rho", "global_log_rms")
CAL_S0 = (0.3, 0.6, 0.9, 1.2)
CAL_INPUT = (0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8)
CAL_RIDGE = (1e-9, 1e-8, 1e-7, 1e-5, 1e-3)
CAL_SEEDS = (70, 71)
PILOT_SEEDS = (100, 101, 102)
CONFIRM_SEEDS = tuple(range(200, 220))


def seed_for(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**32 - 1)


def make_raw_task(task: str, size: int, seed: int):
    """Source task definitions matched to tcr_core, but output a continuous RAW series."""
    rng = np.random.default_rng(seed)
    if task == "memory_d10":
        lag = 10
        series = rng.uniform(-1.0, 1.0, size + lag + 1)
        return series[lag:lag+size], series[:size]
    if task == "narma10":
        order, warm = 10, 160
        full_size = size + warm + order + 2
        for _ in range(50):
            u = rng.uniform(0.0, 0.5, full_size)
            y = np.zeros(full_size)
            stable = True
            for t in range(order, full_size-1):
                y[t+1] = (0.3*y[t] + 0.05*y[t]*np.sum(y[t-order+1:t+1])
                          + 1.5*u[t-order+1]*u[t] + 0.1)
                if not np.isfinite(y[t+1]) or abs(y[t+1]) > 100:
                    stable = False
                    break
            if stable and np.std(y[warm+order:warm+order+size]) > 1e-10:
                start = warm+order
                return 2.0*(u[start:start+size]-0.25), y[start:start+size]
        raise RuntimeError("NARMA generation failed")
    if task == "lorenz_x":
        dt, warm = 0.01, 1200
        total = size + warm + 1
        state = np.array([1.0+0.01*rng.normal(), 1.0+0.01*rng.normal(),
                          1.0+0.01*rng.normal()])
        xs = np.empty(total)
        for t in range(total):
            xs[t] = state[0]
            x,y,z = state
            state = state + dt*np.array([10.0*(y-x), x*(28.0-z)-y,
                                         x*y - (8.0/3.0)*z])
        series = xs[warm:warm+size+1]
        return series[:-1], series[1:]
    raise ValueError(task)


def make_data(task, case_seed, lengths, washout):
    n = washout + sum(lengths)
    raw_u, raw_y = make_raw_task(task, n, seed_for("data", task, case_seed))
    train_slice = slice(washout, washout+lengths[0])
    mu_u, sd_u = np.mean(raw_u[train_slice]), np.std(raw_u[train_slice])
    mu_y, sd_y = np.mean(raw_y[train_slice]), np.std(raw_y[train_slice])
    assert sd_u > 1e-12 and sd_y > 1e-12
    u0 = (raw_u-mu_u)/sd_u
    y = (raw_y-mu_y)/sd_y
    U = np.stack([u0, np.zeros_like(u0)],axis=1)
    s0 = washout
    slices = (slice(s0,s0+lengths[0]),
              slice(s0+lengths[0],s0+lengths[0]+lengths[1]),
              slice(s0+lengths[0]+lengths[1],n))
    return U, y, slices


def make_base(N, task, case_seed):
    rng = np.random.default_rng(seed_for("substrate",N,task,case_seed))
    scores = rng.normal(size=(N,N))
    np.fill_diagonal(scores, -1e9)
    signs = rng.choice(np.array([-1.0,1.0]), size=(N,N))
    np.fill_diagonal(signs,0.0)
    Win_unit = rng.uniform(-1,1,size=(N,2))
    return scores, signs, Win_unit


def norm_rms(M, s0):
    N = len(M)
    rms = np.sqrt(np.mean(np.square(M)))
    return M*(s0/(np.sqrt(N)*rms))


def spectral_radius(W):
    vals = np.linalg.eigvals(W)
    return float(np.max(np.abs(vals)))


def matrices(base, taus, s0):
    scores, signs, _ = base
    A = []
    C = []
    for tau in taus:
        logits = scores/tau
        # Fixed raw scores, excluded diagonal; stable row-normalized softmax.
        row_z = np.exp(logits - np.max(logits,axis=1,keepdims=True))
        alloc = row_z / np.sum(row_z,axis=1,keepdims=True)
        M = signs*alloc
        A.append(norm_rms(M,s0))
        # Global lognormal magnitudes, same A and S. Global shift cancels in RMS.
        raw = signs*np.exp(logits - np.max(logits))
        C.append(norm_rms(raw,s0))
    A = np.stack(A)
    C = np.stack(C)
    k_ref = len(taus)//2
    rho_ref = spectral_radius(A[k_ref])
    B = np.stack([W*(rho_ref/spectral_radius(W)) for W in A])
    assert np.allclose(A[k_ref], B[k_ref],rtol=2e-12,atol=2e-12)
    # A and C have the same exact row-relative magnitude allocations.
    pa = np.abs(A)/np.sum(np.abs(A),axis=2,keepdims=True)
    pc = np.abs(C)/np.sum(np.abs(C),axis=2,keepdims=True)
    assert np.allclose(pa,pc,rtol=1e-10,atol=2e-12)
    return {"row_rms":A,"row_rho":B,"global_log_rms":C},rho_ref


def simulate(Ws, Win, U):
    """Run washout and all segments in causal order without state reset."""
    Ws = np.asarray(Ws)
    K,N,_ = Ws.shape
    state = np.zeros((K,N),dtype=float)
    hist = np.empty((len(U),K,N),dtype=float)
    for t,u in enumerate(U):
        state = np.tanh((Ws@state[:,:,None])[:,:,0] + (Win@u)[None,:])
        hist[t] = state
    return hist


def fit_predict(X, target, fit_slice, predict_slice, ridge):
    Xtrain = X[fit_slice]
    ytrain = target[fit_slice]
    bias = np.ones((len(Xtrain),1))
    Xb = np.column_stack([Xtrain,bias])
    beta = np.linalg.solve(Xb.T@Xb + ridge*np.eye(Xb.shape[1]), Xb.T@ytrain)
    Xt = X[predict_slice]
    return np.column_stack([Xt,np.ones((len(Xt),1))])@beta


def nrmse(y,pred):
    # y already normalized using *training target* standard deviation.
    return float(np.sqrt(np.mean((y-pred)**2)))


def cal_grid(smoke=False):
    if smoke:
        return [(s,inp,r) for s in (0.6,1.0) for inp in (0.2,0.8)
                for r in (1e-5,1e-3)]
    return [(s,inp,r) for s in CAL_S0 for inp in CAL_INPUT
            for r in CAL_RIDGE]


def calibrate(tasks,N,lengths,washout,taus,smoke=False):
    """Same equal-budget reference-T search separately for row RMS and lognormal.

    Each candidate is averaged over the SAME two calibration data/substrate seeds.
    Ridge changes only the readout and does not require replaying reservoir states.
    """
    seeds=(70,) if smoke else CAL_SEEDS
    grid=cal_grid(smoke)
    s_vals=sorted(set(g[0] for g in grid))
    u_vals=sorted(set(g[1] for g in grid))
    r_vals=sorted(set(g[2] for g in grid))
    tref=float(taus[len(taus)//2])
    rows=[]
    best={}
    for task in tasks:
        print(f"calibration: task={task}, grid={len(grid)} per family, "
              f"seeds={len(seeds)}, T_ref={tref:.7f}",flush=True)
        for seed in seeds:
            U,y,splits=make_data(task,seed,lengths,washout)
            base=make_base(N,task,seed)
            for s0 in s_vals:
                matrices_at_ref,_=matrices(base,np.array([tref]),s0)
                for fam in ("row_rms","global_log_rms"):
                    W=matrices_at_ref[fam][0]
                    for scale in u_vals:
                        state=simulate(W[None],base[2]*scale,U)[:,0,:]
                        for ridge in r_vals:
                            pred=fit_predict(state,y,splits[0],splits[1],ridge)
                            rows.append({"task":task,"seed":seed,"family":fam,
                                         "s0":float(s0),"input_scale":float(scale),
                                         "ridge":float(ridge),
                                         "val_nrmse":nrmse(y[splits[1]],pred)})
        task_df=pd.DataFrame(x for x in rows if x["task"]==task)
        task_best={}
        for fam in ("row_rms","global_log_rms"):
            g=(task_df[task_df.family==fam]
               .groupby(["s0","input_scale","ridge"],as_index=False)
               .agg(val_nrmse=("val_nrmse","mean"),sd_cal_seeds=("val_nrmse","std")))
            v=g.sort_values(["val_nrmse","s0","input_scale","ridge"],
                            kind="stable").iloc[0]
            selected={"s0":float(v.s0),"input_scale":float(v.input_scale),
                      "ridge":float(v.ridge),
                      "calibration_val_nrmse":float(v.val_nrmse),
                      "calibration_sd_seeds":float(v.sd_cal_seeds) if np.isfinite(v.sd_cal_seeds) else None,
                      "at_lower_input_boundary":bool(np.isclose(v.input_scale,min(u_vals))),
                      "at_lower_ridge_boundary":bool(np.isclose(v.ridge,min(r_vals),rtol=1e-12,atol=0)),
                      "at_s0_boundary":bool(np.isclose(v.s0,min(s_vals)) or np.isclose(v.s0,max(s_vals)))}
            task_best[fam]=selected
        best[task]=task_best
        print(f"  row: {task_best['row_rms']}",flush=True)
        print(f"  log: {task_best['global_log_rms']}",flush=True)
    return best,pd.DataFrame(rows)


def descriptors(W):
    absolute=np.abs(W)
    rows=absolute.sum(axis=1)
    rel=absolute/rows[:,None]
    entropy=-np.sum(np.where(rel>0, rel*np.log(np.maximum(rel,1e-300)),0),axis=1)
    eig=np.linalg.eigvals(W)
    fro2=float(np.sum(W**2))
    henrici=float(np.sqrt(max(0.0,fro2-np.sum(np.abs(eig)**2)))/np.sqrt(fro2))
    return {"effective_support":float(np.mean(np.exp(entropy))),
            "row_entropy":float(np.mean(entropy)),
            "row_mass_cv":float(np.std(rows)/np.mean(rows)),
            "spectral_radius":float(np.max(np.abs(eig))),
            "spectral_norm":float(np.linalg.norm(W,ord=2)),
            "henrici":henrici}


def evaluate_seed(task,seed,N,lengths,washout,taus,hyper):
    """Primary A/B/C shared task-wise calibration only."""
    U,y,splits=make_data(task,seed,lengths,washout)
    base=make_base(N,task,seed)
    shared=hyper["row_rms"]
    banks,rho_ref=matrices(base,taus,shared["s0"])
    stack=np.concatenate([banks[arm] for arm in ARMS],axis=0)
    states=simulate(stack,base[2]*shared["input_scale"],U)
    records=[]
    for aidx,arm in enumerate(ARMS):
        for j,tau in enumerate(taus):
            i=aidx*len(taus)+j
            pv=fit_predict(states[:,i,:],y,splits[0],splits[1],shared["ridge"])
            pt=fit_predict(states[:,i,:],y,splits[0],splits[2],shared["ridge"])
            records.append({"task":task,"seed":seed,"N":N,"arm":arm,
                            "comparison":"shared_hyperparameters",
                            "temperature_idx":j,"tau":float(tau),
                            "tau_ref":float(taus[len(taus)//2]),"rho_ref":rho_ref,
                            "calibration_s0":shared["s0"],
                            "calibration_input_scale":shared["input_scale"],
                            "calibration_ridge":shared["ridge"],
                            "val_nrmse":nrmse(y[splits[1]],pv),
                            "test_nrmse":nrmse(y[splits[2]],pt),
                            **descriptors(banks[arm][j])})
    return records


def analyses(df):
    summary=[]
    for (task,seed,arm),grp in df.groupby(["task","seed","arm"],sort=False):
        g=grp.sort_values("temperature_idx")
        idx=int(np.argmin(g.val_nrmse.to_numpy()))
        summary.append({"task":task,"seed":seed,"arm":arm,
                        "sd_test_curve":float(np.std(g.test_nrmse.to_numpy())),
                        "val_selected_tau":float(g.tau.iloc[idx]),
                        "val_selected_test_nrmse":float(g.test_nrmse.iloc[idx]),
                        "reference_test_nrmse":float(g.test_nrmse.iloc[len(g)//2]),
                        "test_oracle_tau_descriptive":float(g.tau.iloc[int(np.argmin(g.test_nrmse.to_numpy()))])})
    return pd.DataFrame(summary)


def assert_dataset_invariants(tasks,N,lengths,washout,taus,hyper):
    for task in tasks:
        U,y,splits=make_data(task,82,lengths,washout)
        assert U.shape==(washout+sum(lengths),2)
        assert abs(float(np.mean(U[splits[0],0])))<1e-10
        assert abs(float(np.std(U[splits[0],0]))-1)<1e-10
        assert abs(float(np.mean(y[splits[0]])))<1e-10
        assert abs(float(np.std(y[splits[0]]))-1)<1e-10
        base=make_base(N,task,82)
        banks,_=matrices(base,taus,hyper[task]["row_rms"]["s0"])
        ref=len(taus)//2
        assert np.max(np.abs(banks["row_rms"][ref]-banks["row_rho"][ref]))<1e-10
        for W in banks["row_rho"]:
            assert np.isclose(spectral_radius(W),spectral_radius(banks["row_rms"][ref]),rtol=1e-7)
        for rowW,logW in zip(banks["row_rms"],banks["global_log_rms"]):
            assert np.allclose(np.abs(rowW)/np.abs(rowW).sum(axis=1,keepdims=True),
                               np.abs(logW)/np.abs(logW).sum(axis=1,keepdims=True),
                               atol=1e-10)
        states=simulate(banks["row_rms"][ref:ref+1],
                        base[2]*hyper[task]["row_rms"]["input_scale"],U)
        assert states.shape[0]>washout and np.any(np.abs(states[washout-1])>1e-9)
    return True


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["smoke","calibrate","pilot","confirm"],required=True)
    p.add_argument("--calibration",default=None,
                   help="Frozen v2 calibration.json from v2, required for pilot/confirm")
    p.add_argument("--output",default=None)
    p.add_argument("--confirm-frozen",action="store_true",
                   help="Required for confirmation after the researcher freezes v2.1 protocol")
    args=p.parse_args()
    if args.stage=="confirm" and not args.confirm_frozen:
        p.error("Confirmation requires explicit --confirm-frozen after approving NC-ABL-v2.1")
    smoke=args.stage=="smoke"
    N=20 if smoke else 60
    taus=np.geomspace(TAU_MIN,TAU_MAX,5 if smoke else 13)
    lengths=(160,80,80) if smoke else (1200,500,500)
    washout=20 if smoke else 100
    tasks=("memory_d10",) if smoke else FULL_TASKS
    out=Path(args.output or (Path(__file__).parent/("results_"+args.stage)))
    out.mkdir(parents=True,exist_ok=True)
    if args.stage in ("calibrate","smoke"):
        hyper,cal_df=calibrate(tasks,N,lengths,washout,taus,smoke=smoke)
        cal_df.to_csv(out/"calibration_grid.csv",index=False)
        configuration={"protocol":"NC-ABL-v2","stage":"calibrate" if not smoke else "smoke",
                       "per_task":hyper,"N":N,"taus":taus.tolist(),
                       "lengths":list(lengths),"washout":washout,"tasks":list(tasks),
                       "calibration_seed_ids":list((70,) if smoke else CAL_SEEDS),
                       "candidate_s0":list((0.6,1.0) if smoke else CAL_S0),
                       "candidate_input_scale":list((0.2,0.8) if smoke else CAL_INPUT),
                       "candidate_ridge":list((1e-5,1e-3) if smoke else CAL_RIDGE)}
        write_json(out/"calibration.json",configuration)
        if args.stage=="calibrate":
            print("Calibration-only completed. Examine results before pilot/confirmation.")
            print("Files:",out)
            return
    else:
        if not args.calibration:
            p.error("--calibration results_calibrate/calibration.json is required")
        configuration=json.loads(Path(args.calibration).read_text())
        if configuration.get("protocol")!="NC-ABL-v2" or configuration.get("stage")!="calibrate":
            p.error("v2 calibration-only checkpoint required")
        if configuration.get("N")!=N or not np.allclose(configuration.get("taus"),taus):
            p.error("calibration grid or reservoir-size mismatch")
        if list(configuration.get("tasks",[]))!=list(tasks):
            p.error("calibration task mismatch")
        if configuration.get("lengths")!=list(lengths) or configuration.get("washout")!=washout:
            p.error("calibration data-length mismatch")
        hyper=configuration["per_task"]
    assert_dataset_invariants(tasks,N,lengths,washout,taus,hyper)
    if args.stage=="smoke":
        eval_seeds=(80,)
    elif args.stage=="pilot":
        eval_seeds=PILOT_SEEDS
    else:
        eval_seeds=CONFIRM_SEEDS
    manifest={"protocol":"NC-ABL-v2.1","stage":args.stage,
              "N":N,"taus":taus.tolist(),"tasks":list(tasks),
              "evaluation_seed_ids":list(eval_seeds),
              "hyperparameters":hyper,
              "calibration_sha256":(hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()
                                     if args.calibration else None),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "primary_arms":list(ARMS),
              "comparison_design":"all arms share row_rms reference calibration"}
    write_json(out/"run_manifest.json",manifest)
    rows=[]
    for task in tasks:
        for seed in eval_seeds:
            print(f"{args.stage}: {task}, seed {seed}",flush=True)
            rows.extend(evaluate_seed(task,seed,N,lengths,washout,taus,hyper[task]))
    df=pd.DataFrame(rows)
    df.to_csv(out/"candidate_metrics.csv",index=False)
    cases=analyses(df)
    cases.to_csv(out/"case_metrics.csv",index=False)
    print("Reference temperature:",taus[len(taus)//2])
    print("Chosen per-task calibration:",json.dumps(hyper,indent=2))
    print("Results:",out)
    print("Mean validation-selected test NRMSE (descriptive):")
    print(cases.groupby(["task","arm"])["val_selected_test_nrmse"].mean().to_string())

if __name__=="__main__":
    main()
