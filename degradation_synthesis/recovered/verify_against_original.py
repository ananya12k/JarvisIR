"""Compare the rebuilt engine against surviving ORIGINAL-engine output
(edfog_mor_ladder test/03, 6 configs, rendered by the purged engine).
Usage: python verify_against_original.py [n_frames]"""
import importlib.util,cv2,numpy as np,glob,os,sys,json
R='/scratch/Ananya_Kulkarni/VISIA_old/inference_out/v6_supplementary_all/02_mor_sensitivity/edfog_mor_ladder/test'
OUT=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location('fs','/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis/fog_synthesis.py'); fs=importlib.util.module_from_spec(spec); spec.loader.exec_module(fs)
os.chdir('/scratch/Ananya_Kulkarni/Video-Depth-Anything'); sys.path.insert(0,'.')
vda=fs.VDAWrapper('vits','/scratch/Ananya_Kulkarni/Video-Depth-Anything',0)
N=int(sys.argv[1]) if len(sys.argv)>1 else 60
gts=sorted(glob.glob(R+'/GT/03/*.png'))[:N]
clips=sorted(glob.glob(R+'/Haze/03_cfg*'))
metas=[json.load(open(c+'/meta.json')) for c in clips]
engs=[fs.PhysicalFogEngine(beta_base=m['beta_base'],airlight_color=m['airlight_color'],fog_layer_m=m['fog_layer_m'],
      heterogeneity=m['heterogeneity'],fbm_seed=m['fbm_seed'],speed_kmh=m['speed_kmh'],fps=m['fps']) for m in metas]
psnr=lambda a,b:10*np.log10(255**2/max(((a.astype(float)-b)**2).mean(),1e-9))
stats={i:dict(psnr=[],mae=[],orig_mean=[],mine_mean=[],orig_sky=[],mine_sky=[],orig_road=[],mine_road=[],orig_std=[],mine_std=[],ident_orig=[],ident_mine=[]) for i in range(len(clips))}
save_idx=N-1
for idx,J,d in fs.iter_scene_depth(vda,gts):
    stem=os.path.basename(gts[idx])
    h,w=J.shape[:2]
    for i,(c,e) in enumerate(zip(clips,engs)):
        O=cv2.imread(os.path.join(c,stem)); M=e.apply_fog(J,d); s=stats[i]
        s['psnr'].append(psnr(O,M)); s['mae'].append(np.abs(O.astype(float)-M).mean())
        for k,X in (('orig',O),('mine',M)):
            g=cv2.cvtColor(X,cv2.COLOR_BGR2GRAY).astype(float)
            s[k+'_mean'].append(g.mean()); s[k+'_std'].append(g.std())
            s[k+'_sky'].append(g[:h//8].mean()); s[k+'_road'].append(g[-h//6:, w//4:3*w//4].mean())
            s['ident_'+k].append((X==J).all(-1).mean())
        if idx==save_idx and i in (4,0,5):
            cv2.imwrite(f'{OUT}/cmp_cfg{i}.jpg',np.hstack([J,O,M,cv2.applyColorMap(np.clip(np.abs(O.astype(int)-M).mean(-1)*4,0,255).astype(np.uint8),cv2.COLORMAP_JET)]))
gJ=np.mean([cv2.cvtColor(cv2.imread(g),cv2.COLOR_BGR2GRAY).mean() for g in gts])
print(f'clean mean gray {gJ:.1f}')
print('cfg  MOR  A    H  | PSNR  MAE | mean o/m    | std o/m     | sky o/m      | road o/m     | ident o/m')
for i,m in enumerate(metas):
    s={k:np.mean(v) for k,v in stats[i].items()}
    print(f"{i}  {m['mor']:4.0f} {m['airlight_color'][0]:.2f} {m['fog_layer_m']:2.0f} | {s['psnr']:.1f} {s['mae']:5.1f} | {s['orig_mean']:5.1f}/{s['mine_mean']:5.1f} | {s['orig_std']:5.1f}/{s['mine_std']:5.1f} | {s['orig_sky']:5.1f}/{s['mine_sky']:5.1f} | {s['orig_road']:5.1f}/{s['mine_road']:5.1f} | {s['ident_orig']:.3f}/{s['ident_mine']:.3f}")
