import h5py
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=240)


class TrainingResult:
    def __init__(self, readout, cache, metadata, experiment_dir, feature_config, scaler_params, feature_info=None):
        self.readout = readout
        self.cache = cache
        self.meta = metadata
        self.feature_config = feature_config
        self.scaler_params = scaler_params
        self.feature_info = feature_info
        
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
            
            if self.feature_info is not None and self.feature_info:
                # Store feature mapping in a structured, matrix-based format
                first_feature = self.feature_info[0]
                feature_type = first_feature.get('type', 'unknown')
                
                # Dynamically get the column keys (e.g., 'node_id', 'dim') from the dict
                matrix_keys = sorted([k for k in first_feature.keys() if k != 'type'])
                
                # Create a matrix where each row corresponds to a feature
                feature_map_matrix = np.array([[fi[k] for k in matrix_keys] for fi in self.feature_info], dtype=np.int32)
                
                # Save to a dedicated group for clarity
                f_group = mg.create_group('feature_map')
                f_group.attrs['type'] = feature_type
                f_group.attrs['keys'] = json.dumps(matrix_keys)
                f_group.create_dataset('dof_map', data=feature_map_matrix)
            else:
                # Fallback for older feature extractors or if info is missing
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
            
        return self.output_path

class Trainer:
    """
    Standardizes data GLOBALLY before splitting into fixed train/test lengths.
    Structure: [1, X_standardized]
    """
    def __init__(self, features, readout, experiment_dir, loader, washout=5.0, train_duration=20.0, test_duration=5.0):
        self.loader = loader
        self.features = features
        self.readout = readout
        self.experiment_dir = Path(experiment_dir)
        self.washout = washout
        self.train_duration = train_duration
        self.test_duration = test_duration

    def train(self, y_full, task_name="Custom"):
        # 1. Convert time-based lengths to frame-based lengths
        dt = self.loader.dt
        washout_len_frames = int(self.washout / dt)
        train_len_frames = int(self.train_duration / dt)
        test_len_frames = int(self.test_duration / dt)

        # 2. Extract Full Features [T, N]
        X_full = self.features.transform(self.loader)
        
        # 3. Check Lengths
        required_len = washout_len_frames + train_len_frames + test_len_frames
        if len(X_full) < required_len:
            raise ValueError(
                f"Simulation too short! Need {required_len} frames ({self.washout + self.train_duration + self.test_duration:.2f}s), "
                f"but simulation only has {len(X_full)} frames ({len(X_full)*dt:.2f}s)."
            )

        # 4. Standardize X and y
        scaler_X = StandardScaler()
        X_std = scaler_X.fit_transform(X_full)
        
        scaler_y = StandardScaler()
        y_std = scaler_y.fit_transform(y_full.reshape(-1, 1)).flatten()

        # 5. Construct Design Matrix L = [1, X_std]
        ones = np.ones((X_std.shape[0], 1))
        L_full = np.hstack((ones, X_std))

        # 6. Slicing
        start_train = washout_len_frames
        end_train   = washout_len_frames + train_len_frames
        start_test  = end_train
        end_test    = end_train + test_len_frames

        L_train = L_full[start_train:end_train]
        y_train = y_std[start_train:end_train]

        L_test  = L_full[start_test:end_test]
        y_test  = y_std[start_test:end_test]

        print(f"Training with {train_len_frames} frames ({self.train_duration:.2f}s) | "
              f"Testing with {test_len_frames} frames ({self.test_duration:.2f}s)")

        # 7. Fit Readout
        self.readout.fit(L_train, y_train)
        
        # 8. Store Results
        cache = {
            "train": (L_train, y_train, self.readout.predict(L_train)),
            "test":  (L_test, y_test, self.readout.predict(L_test))
        }
        
        meta = {'sim': str(self.loader.sim_path), 'task': task_name}
        feature_config = {
            "type": self.features.__class__.__name__, 
            "washout_duration": self.washout,
            "train_duration": self.train_duration,
            "test_duration": self.test_duration
        }
        
        scaler_params = {
            'X_mean': scaler_X.mean_, 'X_scale': scaler_X.scale_,
            'y_mean': scaler_y.mean_, 'y_scale': scaler_y.scale_
        }
        
        feature_info = self.features.get_feature_info(self.loader)
        
        return TrainingResult(self.readout, cache, meta, self.experiment_dir, feature_config, scaler_params, feature_info)