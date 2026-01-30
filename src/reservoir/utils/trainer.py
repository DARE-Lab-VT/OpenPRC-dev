import h5py
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=240)


class TrainingResult:
    def __init__(self, readout, cache, metadata, experiment_dir, feature_config, scaler_params):
        self.readout = readout
        self.cache = cache
        self.meta = metadata
        self.feature_config = feature_config
        self.scaler_params = scaler_params
        
        safe_task_name = self.meta['task'].lower()
        self.output_path = Path(experiment_dir) / "readout" / f"readout_{safe_task_name}.h5"

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving Readout Artifact to: {self.output_path}")
        
        with h5py.File(self.output_path, 'w') as f:
            f.attrs['task_name'] = self.meta['task']
            f.attrs['source_simulation'] = str(self.meta['sim'])
            f.attrs['training_timestamp'] = datetime.now().isoformat()
            f.attrs['feature_config'] = json.dumps(self.feature_config)
            
            # Save Scaler Stats
            f.create_dataset('preprocessing/X_mean', data=self.scaler_params['X_mean'])
            f.create_dataset('preprocessing/X_scale', data=self.scaler_params['X_scale'])
            f.create_dataset('preprocessing/y_mean', data=self.scaler_params['y_mean'])
            f.create_dataset('preprocessing/y_scale', data=self.scaler_params['y_scale'])
            
            # Model
            mg = f.create_group('model')
            mg.create_dataset('weights', data=self.readout.weights)
            mg.create_dataset('regularization', data=getattr(self.readout.model, 'alpha', 0.0))
            mg.create_dataset('feature_indices', data=np.arange(self.cache['train'][0].shape[1]))

            # Training Data
            tg = f.create_group('training')
            tg.create_dataset('input', data=self.cache['train'][0])
            tg.create_dataset('target', data=self.cache['train'][1])
            tg.create_dataset('prediction', data=self.cache['train'][2])
            
            # Test Data (Mapped to 'validation' group for schema compliance)
            vg = f.create_group('validation')
            vg.create_dataset('input', data=self.cache['test'][0])
            vg.create_dataset('target', data=self.cache['test'][1])
            vg.create_dataset('prediction', data=self.cache['test'][2])
            vg.create_group('metrics')
            
        return self.output_path

class Trainer:
    """
    Standardizes data GLOBALLY before splitting into fixed train/test lengths.
    Structure: [1, X_standardized]
    """
    def __init__(self, features, readout, experiment_dir, washout=500, train_len=2000, test_len=500, actuator_idx=0):
        self.features = features
        self.readout = readout
        self.experiment_dir = Path(experiment_dir)
        self.washout = washout
        self.train_len = train_len
        self.test_len = test_len
        self.actuator_idx = actuator_idx

    def fit(self, state_loader, task):
        # 1. Extract Full Features [T, N]
        X_full = self.features.transform(state_loader)
        
        # 2. Check Lengths
        required_len = self.washout + self.train_len + self.test_len
        if len(X_full) < required_len:
            raise ValueError(
                f"Simulation too short! Need {required_len} frames, got {len(X_full)}."
            )

        # 3. Handle task type for metadata and full target array
        if isinstance(task, np.ndarray):
            task_name = "Custom"
            y_full = task
        else:
            task_name = task.__class__.__name__
            y_full = task.generate(state_loader.get_actuation_signal(actuator_idx=self.actuator_idx)).flatten()

        # 4. GLOBAL STANDARDIZATION (Before Splitting)
        # Standardize X
        scaler_X = StandardScaler()
        X_std = scaler_X.fit_transform(X_full)
        
        # Standardize y
        scaler_y = StandardScaler()
        y_std = scaler_y.fit_transform(y_full.reshape(-1, 1)).flatten()

        # 5. Construct Design Matrix L = [1, X_std]
        ones = np.ones((X_std.shape[0], 1))
        L_full = np.hstack((ones, X_std))

        # 6. Slicing (Fixed Integers)
        start_train = self.washout
        end_train   = self.washout + self.train_len
        start_test  = end_train
        end_test    = end_train + self.test_len

        # Train Slice
        L_train = L_full[start_train:end_train]
        y_train = y_std[start_train:end_train]

        # Test Slice
        L_test  = L_full[start_test:end_test]
        y_test  = y_std[start_test:end_test]

        print(f"Training: {L_train.shape} | Testing: {L_test.shape}")

        # 7. Fit Readout
        # fit_intercept=False is required in Readout because we added the bias column
        self.readout.fit(L_train, y_train)
        
        # 8. Store Results
        cache = {
            "train": (L_train, y_train, self.readout.predict(L_train)),
            "test":  (L_test, y_test, self.readout.predict(L_test))
        }
        
        meta = {'sim': state_loader.sim_path, 'task': task_name}
        feature_config = {
            "type": self.features.__class__.__name__, 
            "washout": self.washout,
            "train_len": self.train_len,
            "test_len": self.test_len
        }
        
        # Store both scalers for full reproducibility
        scaler_params = {
            'X_mean': scaler_X.mean_, 'X_scale': scaler_X.scale_,
            'y_mean': scaler_y.mean_, 'y_scale': scaler_y.scale_
        }
        
        return TrainingResult(self.readout, cache, meta, self.experiment_dir, feature_config, scaler_params)