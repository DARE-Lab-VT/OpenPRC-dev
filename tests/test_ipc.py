"""Analytical and regression checks for clean-IID IPC; no physics/GPU required."""
import unittest
import numpy as np
from openprc.analysis.utils.training_utils import (
    compute_ipc_components as ipc, compute_ipc_components_gpu as gpu,
    scale_iid_input, legendre_target, aggregate_ipc_heatmaps)


class IPCChecks(unittest.TestCase):
    def setUp(self):
        self.u=np.random.default_rng(812).uniform(-1,1,12000)

    def test_iid_generation_and_file_override(self):
        import tempfile
        from pathlib import Path
        import h5py
        from openprc.examples.run_plot_heatmap import load_or_generate_iid
        args=(None, 'iid', 'numpy_randomstate', 42, 30., 120., (-1.,1.))
        values=load_or_generate_iid(*args)
        self.assertEqual(len(values),3600)
        np.testing.assert_array_equal(values,np.random.RandomState(42).uniform(-1,1,3600))
        np.testing.assert_array_equal(values,load_or_generate_iid(*args))
        pcg=load_or_generate_iid(None,'iid','numpy_pcg64',42,30.,120.,(-1.,1.))
        np.testing.assert_array_equal(pcg,np.random.Generator(np.random.PCG64(42)).uniform(-1,1,3600))
        self.assertFalse(np.array_equal(values,pcg))
        with tempfile.TemporaryDirectory() as folder:
            for extension in ('.npy','.h5'):
                path=Path(folder)/('symbols'+extension)
                original=np.array([[-.4],[.2],[.6]])
                if extension=='.npy': np.save(path,original)
                else:
                    with h5py.File(path,'w') as f: f['iid']=original
                loaded=load_or_generate_iid(path,'iid','unused',-1,30.,120.,(-1.,1.))
                np.testing.assert_array_equal(loaded,original[:,0])
        partial=load_or_generate_iid(None,'iid','numpy_randomstate',42,30.,.11,(-1.,1.))
        self.assertEqual(len(partial),4)
        saved=load_or_generate_iid(None,'iid','numpy_pcg64',42,1/.033,30.,(-1.,1.))
        self.assertEqual(len(saved),910)
        with self.assertRaises(ValueError):
            load_or_generate_iid(None,'iid','numpy_randomstate',42,30.,0.,(-1.,1.))

    def test_delay_bank_and_independent_targets(self):
        u=self.u
        x=np.column_stack([np.roll(u,j) for j in range(3)])
        _,score,e=ipc(x,u,4,2,10,6000,6000,1,ridge=0.)
        for lag in range(5):
            target=np.zeros(5,int);target[lag]=1
            value=score[np.all(e==target,axis=1)][0]
            if lag<3:self.assertGreater(value,.999999)
            else:self.assertLess(value,.01)
        self.assertLess(score[e.sum(1)==2].max(),.01)
        self.assertAlmostEqual(score[e.sum(1)==1].sum(),3.,delta=.02)

    def test_known_nonlinear_product(self):
        y=legendre_target(self.u,[1,1])
        x=np.nan_to_num(y)[:,None]
        _,scores,e=ipc(x,self.u,1,2,10,6000,6000,1,ridge=0.)
        self.assertGreater(scores[np.all(e==[1,1],axis=1)][0],.999999)
        self.assertLess(scores[e.sum(1)==1].max(),.01)

    def test_population_bounds_not_extrema(self):
        np.testing.assert_allclose(scale_iid_input([.2,.4,.7],(0.,1.)),[-.6,-.2,.4])
        x=self.u[:,None]
        a=ipc(x,self.u,2,2,5,6000,6000,1)[1]
        b=ipc(x,(self.u+1)/2,2,2,5,6000,6000,1,input_bounds=(0.,1.))[1]
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_common_history_window(self):
        x=np.column_stack((self.u,np.roll(self.u,1)))
        a=ipc(x,self.u,4,2,0,6000,6000,1)[1]
        b=ipc(x,self.u,4,2,4,6000,6000,1)[1]
        np.testing.assert_array_equal(a,b)

    def test_raw_scores_and_no_default_cutoff(self):
        rng=np.random.default_rng(99)
        x=rng.normal(size=(len(self.u),4))
        raw=ipc(x,self.u,3,2,10,6000,6000,1,return_raw=True)[1]
        clipped=ipc(x,self.u,3,2,10,6000,6000,1)[1]
        self.assertTrue(np.any(raw<0))
        np.testing.assert_array_equal(clipped,np.maximum(raw,0))

    def test_mismatch_and_invalid_windows_rejected(self):
        x=self.u[:,None]
        for kwargs in ({'input_bounds':(0,1)}, {'ridge':-1}):
            with self.assertRaises(ValueError):ipc(x,self.u,2,2,5,6000,6000,1,**kwargs)
        with self.assertRaises(ValueError):ipc(x[:-1],self.u,2,2,5,6000,6000,1)
        with self.assertRaises(ValueError):ipc(x,self.u,20,2,0,10,100,1)
        with self.assertRaises(ValueError):ipc(x,self.u,2,2,5,6000,6001,1)

    def test_obsolete_arguments_rejected(self):
        args=(self.u[:,None],self.u,2,2,5,6000,6000,1)
        for name in ('epsilon', 'interp_factor'):
            with self.assertRaises(TypeError):
                ipc(*args,**{name:1})
        with self.assertRaises(TypeError):
            gpu(*args,epsilon=0.)
        from openprc.analysis.tasks.imitation import memory_task
        with self.assertRaises(TypeError):
            memory_task(self.u[:,None],self.u,5,6000,6000,2,2,eps=0.)
        from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
        with self.assertRaises(TypeError):
            MemoryBenchmark().run(None,self.u,eps=0.)
        # Old positional options must not silently become a different setting.
        with self.assertRaises(TypeError):
            ipc(*args,1,0.)

    def test_aggregation_definitions(self):
        e=np.array([[1,0],[0,1],[2,0],[1,1],[0,2]])
        exact,cum=aggregate_ipc_heatmaps([1,.5,-.1,.2,0],e)
        np.testing.assert_allclose(exact,[[1,.5],[0,.2]])
        np.testing.assert_allclose(cum,[[1,.75],[.5,1.7/5]])

    def test_benchmark_saves_legendre_not_monomial_targets(self):
        import tempfile
        from types import SimpleNamespace
        from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
        captured=[]
        u=(self.u[:400]+1)/2
        features=SimpleNamespace(transform=lambda loader:u[:,None])
        def train(y,task_name):
            captured.append(y[:,0])
            return SimpleNamespace(save=lambda:None)
        with tempfile.TemporaryDirectory() as folder:
            trainer=SimpleNamespace(experiment_dir=folder,features=features,
                loader=SimpleNamespace(dt=1.),washout=10.,train_duration=190.,
                test_duration=200.,train=train)
            result=MemoryBenchmark().run(trainer,u,tau_s=1,n_s=2,k_delay=1,
                input_bounds=(0.,1.),save_readouts_for=['P2(u(t-0))'])
            expected=legendre_target(2*u-1,[2,0])
            np.testing.assert_allclose(captured[0],expected)
            self.assertNotIn('epsilon',result.metadata)
            result.save()

    def test_float32_timestamp_rounding(self):
        from openprc.examples.run_plot_heatmap import multiplex_states
        time=(np.arange(910)*.033).astype(np.float32)
        states=np.arange(910,dtype=float)[:,None]
        result=multiplex_states(time,states,910,1/.033,0.,(0.,))
        self.assertEqual(result.shape,(910,1))
        self.assertEqual(result[-1,0],909.)
        with self.assertRaises(ValueError):
            multiplex_states(time,states,911,1/.033,0.,(0.,))

    def test_noninteger_sampling_alignment(self):
        from openprc.examples.run_plot_heatmap import multiplex_states
        t=np.arange(1200)/119.88
        x=np.column_stack((t,2*t))
        mux=multiplex_states(t,x,500,54.,0.,(0.,.5))
        samples=(np.arange(500)[:,None]+np.array([0.,.5]))/54.
        expected=np.stack((samples,2*samples),axis=-1).reshape(500,4)
        np.testing.assert_allclose(mux,expected,atol=1e-14)
        with self.assertRaises(ValueError):
            multiplex_states(t,x,600,54.,0.,(0.,.5))

    def test_existing_modules_match_direct_ipc_on_symbol_clock(self):
        import tempfile
        from pathlib import Path
        import h5py
        from sklearn.preprocessing import StandardScaler
        from openprc.examples.run_plot_heatmap import multiplex_states
        from openprc.reservoir.features.node_features import NodePositions
        from openprc.reservoir.io.state_loader import StateLoader
        from openprc.reservoir.training.trainer import Trainer
        from openprc.reservoir.readout.ridge import Ridge
        from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
        u=self.u[:1000]
        hz=54.
        time=np.arange(2400)/119.88
        displacement=np.interp(time,np.arange(len(u))/hz,u)
        positions=np.zeros((len(time),2,3))
        positions[:,0,0]=displacement
        positions[:,1,1]=np.roll(displacement,3)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'simulation.h5'
            with h5py.File(path,'w') as f:
                f['time_series/time']=time
                f['time_series/nodes/positions']=positions
            loader=StateLoader(path)
            features=NodePositions(node_ids=[0,1],dims="all")
            extracted=features.transform(loader).reshape(len(time),2,3)[:,:,:2].reshape(len(time),4)
            displacement=extracted-extracted[:1]
            np.testing.assert_array_equal(displacement[0],np.zeros(4))
            raw=multiplex_states(loader.time,displacement,len(u),hz,0.,(0.,.5))
            self.assertEqual(raw.shape,(1000,8))
            first,stop=173,586
            trainer=Trainer(features,Ridge(1e-6),folder,loader,
                washout=first/hz,train_duration=(stop-first)/hz,
                test_duration=(len(u)-stop)/hz)
            result=MemoryBenchmark().run(trainer,u,tau_s=3,n_s=2,k_delay=1,sample_dt=1/hz,prepared_states=raw)
            scaled=StandardScaler().fit(raw[first:stop]).transform(raw)
            expected=ipc(scaled,u,3,2,first,stop,len(u)-stop,1)[1]
            np.testing.assert_allclose(result.metrics['capacities'],expected,atol=1e-12)
            self.assertAlmostEqual(loader.dt,1/119.88)
            self.assertEqual(result.metadata['sample_dt'],1/hz)
            with self.assertRaises(ValueError):
                MemoryBenchmark().run(trainer,u,tau_s=3,n_s=2,k_delay=1,
                    sample_dt=1/hz,prepared_states=raw,save_readouts_for=['P1(u(t-0))'])

    def test_torch_parity(self):
        try: import torch
        except ImportError:self.skipTest('Torch optional')
        x=np.column_stack((self.u,np.roll(self.u,1)))
        args=(x,self.u,3,2,0,6000,6000,1)
        cpu=ipc(*args,return_raw=True)[1]
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            np.testing.assert_allclose(cpu,gpu(*args,device=device,return_raw=True)[1],atol=2e-5,rtol=2e-5)

    def test_wrong_iid_sequence_fails_delay_bank(self):
        x=np.column_stack((self.u,np.roll(self.u,1)))
        unrelated=np.random.default_rng(7).uniform(-1,1,len(self.u))
        self.assertLess(ipc(x,unrelated,2,1,10,6000,6000,1)[1].sum(),.01)


if __name__=='__main__':unittest.main()
