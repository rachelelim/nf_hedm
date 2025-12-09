#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Near-field grain mapping reconstruction script.

Authors: dcp5303, seg246
"""

import argparse
import logging
import sys
import os

import matplotlib.pyplot as plt
import nf_config
import nfutil_REL as nfutil
import numpy as np
import h5py


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=level
    )


def validate_file(filepath, description):
    if not os.path.isfile(filepath):
        logging.error(f"{description} file not found: {filepath}")
        sys.exit(1)


def load_configuration(config_path):
    validate_file(config_path, "Configuration")
    try:
        configuration = nf_config.open_file(config_path)[0]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)
    return configuration

def predict_output_shapes_and_dtypes(configuration):
    # Use config to predict output array shapes/dtypes
    # This uses nfutil.generate_test_coordinates logic, but does NOT run reconstruction

    cross_sectional_dim = configuration.reconstruction.cross_sectional_dimensions
    v_bnds = configuration.reconstruction.desired_vertical_span
    voxel_spacing = configuration.reconstruction.voxel_spacing
    mask_data_file = configuration.reconstruction.tomography.get('mask_filepath', None)
    vertical_motor_position = configuration.reconstruction.tomography.get('vertical_motor_position', 0.0)

    # Generate grid shape (without running full experiment)
    Xs, Ys, Zs, mask, test_coordinates = nfutil.generate_test_coordinates(
        cross_sectional_dim,
        v_bnds,
        voxel_spacing,
        mask_data_file=mask_data_file,
        vertical_motor_position=vertical_motor_position
    )
    grid_shape = Xs.shape

    # Predict dtypes
    dtypes = {
        'grain_map': np.int32,
        'confidence': np.float32,
        'Xs': np.float32,
        'Ys': np.float32,
        'Zs': np.float32,
        'tomo_mask': bool
    }
    shapes = {
        'grain_map': grid_shape,
        'confidence': grid_shape,
        'Xs': grid_shape,
        'Ys': grid_shape,
        'Zs': grid_shape,
        'tomo_mask': grid_shape
    }
    return shapes, dtypes


def check_hdf5_dataset_conflicts(h5_filepath, shapes, dtypes):
    """
    Checks for conflicting datasets in an HDF5 file.

    Parameters
    ----------
    h5_filepath : str
        Path to the HDF5 file.
    shapes : dict
        Dict of dataset shapes.
    dtypes : dict
        Dict of dataset dtypes.

    Returns
    -------
    conflicts : list
        List of dataset names that conflict (exist with different shape or dtype).
    """
    conflicts = []
    if not os.path.isfile(h5_filepath):
        return conflicts  # No file, so no conflicts

    with h5py.File(h5_filepath, 'r') as hf:
        for dset_name in shapes:
            if dset_name in hf:
                existing = hf[dset_name]
                # Compare shape and dtype
                if existing.shape != shapes[dset_name] or existing.dtype != np.dtype(dtypes[dset_name]):
                    conflicts.append(dset_name)
    return conflicts 

def handle_hdf5_conflicts(h5_filepath, conflicts, overwrite=False):
    """
    Handles conflicting datasets in an HDF5 file.

    Parameters
    ----------
    h5_filepath : str
        Path to the HDF5 file.
    conflicts : list
        List of conflicting dataset names.
    overwrite : bool
        If True, delete conflicting datasets. If False, abort.
    """
    if not conflicts:
        return
    if overwrite:
        with h5py.File(h5_filepath, 'a') as hf:
            for dset_name in conflicts:
                logging.warning(f"Overwriting dataset '{dset_name}' in {h5_filepath}")
                del hf[dset_name]
    else:
        logging.error(
            f"Conflicting datasets found in {h5_filepath}: {conflicts}. "
            "Use --overwrite-hdf5 to remove them."
        )
        raise RuntimeError("HDF5 dataset conflict detected.")

def preflight_hdf5_check(experiment, grain_map, confidence_map, Xs, Ys, Zs, mask, overwrite_hdf5=False):
    h5_path = os.path.join(experiment.output_directory, experiment.analysis_name + '_grain_map_data.h5')
    datasets = {
        'grain_map': grain_map,
        'confidence': confidence_map,
        'Xs': Xs,
        'Ys': Ys,
        'Zs': Zs,
        'tomo_mask': mask
    }
    conflicts = check_hdf5_dataset_conflicts(h5_path, datasets)
    handle_hdf5_conflicts(h5_path, conflicts, overwrite=overwrite_hdf5)


def run_reconstruction(configuration, plot=True, layer_index=0, conf_thresh=0.2):
    # Generate experiment and image stack
    experiment, image_stack = nfutil.generate_experiment(configuration)
    controller = nfutil.build_controller(configuration)

    # Generate test coordinates and mask
    Xs, Ys, Zs, mask, test_coordinates = nfutil.generate_test_coordinates(
        experiment.cross_sectional_dimensions,
        experiment.vertical_bounds,
        experiment.voxel_spacing,
        mask_data_file=experiment.mask_filepath,
        vertical_motor_position=experiment.vertical_motor_position
    )

    # Precompute orientation data
    orientation_data = nfutil.precompute_diffraction_data(
        experiment, controller, experiment.exp_maps
    )

    # Test orientations at coordinates
    raw_exp_maps, raw_confidence, raw_idx = nfutil.test_orientations_at_coordinates(
        experiment, controller, image_stack, orientation_data, test_coordinates, refine_yes_no=0
    )

    # Process raw output
    grain_map, confidence_map = nfutil.process_raw_data(
        raw_confidence, raw_idx, Xs.shape, mask=mask.astype(bool), id_remap=experiment.remap
    )

    # Plot results if requested
    if plot:
        if layer_index < 0 or layer_index >= Xs.shape[0]:
            logging.warning(
                f"Layer index {layer_index} out of bounds, using 0.")
            layer_index = 0
        nfutil.plot_ori_map(
            grain_map, confidence_map, Xs, Zs,
            experiment.exp_maps, layer_index,
            experiment.mat[experiment.material_name],
            experiment.remap, conf_thresh
        )

    # Return all data needed for saving
    return {
        "experiment": experiment,
        "grain_map": grain_map,
        "confidence_map": confidence_map,
        "Xs": Xs,
        "Ys": Ys,
        "Zs": Zs,
        "mask": mask,
        "exp_maps": experiment.exp_maps
    }


def save_reconstruction_results(data):
    experiment = data["experiment"]
    grain_map = data["grain_map"]
    confidence_map = data["confidence_map"]
    Xs = data["Xs"]
    Ys = data["Ys"]
    Zs = data["Zs"]
    mask = data["mask"]
    exp_maps = data["exp_maps"]

    # Save processed data (.npz)
    nfutil.save_nf_data(
        experiment.output_directory, experiment.analysis_name,
        grain_map, confidence_map, Xs, Ys, Zs, exp_maps,
        tomo_mask=mask, id_remap=experiment.remap,
        save_type=['npz']
    )

    # Save processed data for Paraview (.h5, .xdmf, IPF colors)
    nfutil.save_nf_data_for_paraview(
        experiment.output_directory, experiment.analysis_name,
        grain_map, confidence_map, Xs, Ys, Zs, exp_maps,
        experiment.mat[experiment.material_name], tomo_mask=mask,
        id_remap=experiment.remap
    )


def main():
    parser = argparse.ArgumentParser(
        description='Near-field reconstruction grain mapping.'
    )
    parser.add_argument('input_file', type=str, help='Input configuration file (YAML)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')
    parser.add_argument('--layer', type=int, default=0, help='Layer index for visualization')
    parser.add_argument('--conf-thresh', type=float, default=0.2, help='Confidence threshold for plotting')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--overwrite-hdf5', action='store_true', help='Overwrite conflicting datasets in HDF5 output')
    args = parser.parse_args()

    
    setup_logging(args.verbose)
    configuration = load_configuration(args.input_file)

    # --- HDF5 preflight check BEFORE running reconstruction ---
    shapes, dtypes = predict_output_shapes_and_dtypes(configuration)
    h5_path = os.path.join(configuration.output_directory,
                          configuration.analysis_name + '_grain_map_data.h5')
    conflicts = check_hdf5_dataset_conflicts(h5_path, shapes, dtypes)
    try:
        handle_hdf5_conflicts(h5_path, conflicts, overwrite=args.overwrite_hdf5)
    except RuntimeError as e:
        logging.error(f"Aborting due to HDF5 dataset conflict: {e}")
        sys.exit(1)


    reconstruction_data = run_reconstruction(
        configuration,
        plot=not args.no_plot,
        layer_index=args.layer,
        conf_thresh=args.conf_thresh
    )

    save_reconstruction_results(reconstruction_data)

if __name__ == '__main__':
    main()

