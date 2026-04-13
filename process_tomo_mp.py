# %% Necessary Dependencies
import argparse
import logging
import os
import gc
import multiprocessing as mp

import numpy as np
import yaml

import nf_config
import nfutil
import tomoutil_REL as tomoutil

from hexrd import instrument


try:
    import matplotlib.pyplot as plt
    matplot = True
except ImportError:
    logging.warning('no matplotlib, debug plotting disabled')
    matplot = False


# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("tomo_processing.log", mode="a"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
_WORKER_CONFIG = None


def create_memmap_array(arr, path):
    mm = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    del mm


def init_reconstruct_worker(config):
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)

    _WORKER_CONFIG["filtered_rad_stack"] = np.memmap(
        config["filtered_rad_stack_path"],
        dtype=np.dtype(config["filtered_rad_stack_dtype"]),
        mode='r',
        shape=tuple(config["filtered_rad_stack_shape"])
    )


def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)


def reconstruct_mask_for_layer(task):
    global _WORKER_CONFIG
    cfg = _WORKER_CONFIG

    ii, layer_row = task

    reconstruction_fbp = tomoutil.tomo_reconstruct_layer(
        cfg["filtered_rad_stack"],
        cross_sectional_dim=2,
        layer_row=layer_row,
        start_tomo_ang=cfg["start_tomo_ang"],
        end_tomo_ang=cfg["end_tomo_ang"],
        y_center=cfg["rot_axis_pos"],
        pixel_size=cfg["pixel_size"],
        bad_frames=cfg["bad_frames"]
    )

    binary_recon = tomoutil.threshold_and_clean_tomo_layer(
        reconstruction_fbp,
        cfg["recon_thresh"] * 2,
        cfg["noise_obj_size"],
        cfg["min_hole_size"],
        erosion_iter=cfg["erosion_iter"],
        dilation_iter=cfg["dilation_iter"]
    )

    tomo_mask = tomoutil.crop_and_rebin_tomo_layer(
        binary_recon,
        cfg["voxel_spacing"],
        cfg["pixel_size"],
        x_center=cfg["x_center"],
        z_center=cfg["z_center"],
        x_length=cfg["x_length"],
        z_length=cfg["z_length"]
    )

    return ii, tomo_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Generate tomography mask')
    parser.add_argument(
        'input_file',
        nargs='?',
        default='tomo_config.yml',
        help='Input config file'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help='Number of worker processes'
    )
    return parser.parse_args()


def resolve_tiff_ext(folder, stem, num_digits, start_idx):
    for ext in ('.tif', '.tiff'):
        fname = os.path.join(folder, f'{stem}{start_idx:0{num_digits}d}{ext}')
        if os.path.exists(fname):
            return ext
    raise FileNotFoundError(
        f'Could not find TIFF file with either .tif or .tiff in {folder}'
    )


def load_tomo_data(tomo):
    filetype = tomo.filetype
    img_stem = tomo.img_stem
    num_digits = tomo.num_digits

    if filetype == 'tiff':
        tomo_data_folder = tomo.tomo_images.folder
        tomo_img_start = tomo.tomo_images.img_start
        ext = resolve_tiff_ext(tomo_data_folder, img_stem,
                               num_digits, tomo_img_start)

        tbf_data_folder = tomo.bright.folder
        tbf_img_start = tomo.bright.img_start
        tbf_num_imgs = tomo.bright.num_imgs

        tdf_data_folder = tomo.dark.folder
        tdf_img_start = tomo.dark.img_start
        tdf_num_imgs = tomo.dark.num_imgs

        tomo_num_imgs = tomo.tomo_images.num_imgs

        tdf = tomoutil.gen_median_image(
            tdf_data_folder, tdf_img_start, tdf_num_imgs,
            stem=img_stem, num_digits=num_digits, ext=ext
        )

        tbf = tomoutil.gen_median_image(
            tbf_data_folder, tbf_img_start, tbf_num_imgs,
            stem=img_stem, num_digits=num_digits, ext=ext
        )

        rad_stack, image_stack, filtered_rad_stack = tomoutil.gen_attenuation_rads(
            tomo_data_folder, tbf, tomo_img_start, tomo_num_imgs,
            stem=img_stem, num_digits=num_digits, tdf=tdf, ext=ext
        )

    elif filetype == 'h5':
        tbf_filename = tomo.bright.filename
        tdf_filename = tomo.dark.filename
        tomo_filename = tomo.tomo_images.filename

        tdf = tomoutil.gen_median_image_h5(tdf_filename)
        tbf = tomoutil.gen_median_image_h5(tbf_filename)
        rad_stack, image_stack, filtered_rad_stack = tomoutil.gen_attenuation_rads_h5(
            tomo_filename, tbf, tdf
        )
        tomo_num_imgs = len(rad_stack)

    else:
        raise ValueError(f'Unsupported tomography filetype: {filetype}')

    return rad_stack, image_stack, filtered_rad_stack, tomo_num_imgs


def main():
    args = parse_args()
    fname = args.input_file

    cfg = nf_config.open_file(fname)[0]

    output_stem = cfg.analysis_name
    output_dir = cfg.output_directory
    det_file = cfg.input_files.detector_file
    tomo = cfg.tomography

    ome_range_deg = tomo.ome_range

    recon_thresh = tomo.processing.recon_thresh
    noise_obj_size = tomo.processing.noise_obj_size
    min_hole_size = tomo.processing.min_hole_size
    erosion_iter = tomo.processing.erosion_iter
    dilation_iter = tomo.processing.dilation_iter

    x_center = tomo.reconstruction.x_center
    z_center = tomo.reconstruction.z_center
    x_length = tomo.reconstruction.x_length
    z_length = tomo.reconstruction.z_length
    voxel_spacing = tomo.reconstruction.voxel_spacing
    v_bnds = tomo.reconstruction.v_bnds

    instr = load_instrument(det_file)
    panel = next(iter(instr.detectors.values()))

    nrows = panel.rows

    pixel_size = panel.pixel_size_row
    rot_axis_pos = panel.tvec[0]
    vert_beam_center = panel.tvec[1]

    vert_points = np.arange(
        v_bnds[0] + voxel_spacing / 2.0, v_bnds[1], voxel_spacing)
    center_layer_row = nrows / 2.0 + vert_beam_center / pixel_size
    rows_to_recon = np.round(
        center_layer_row - vert_points / pixel_size).astype(int)

    rad_stack, image_stack, filtered_rad_stack, tomo_num_imgs = load_tomo_data(
        tomo
    )

    frame_intensity = np.sum(image_stack, axis=(1, 2))
    bad_frames = np.argwhere(
        frame_intensity < np.median(frame_intensity) * 0.95)
    del rad_stack, image_stack

    filtered_rad_stack = filtered_rad_stack.astype(np.float16, copy=False)

    os.makedirs(output_dir, exist_ok=True)
    memmap_path = os.path.join(
        output_dir, f'{output_stem}_filtered_rad_stack.dat')

    create_memmap_array(filtered_rad_stack, memmap_path)

    worker_config = {
        "filtered_rad_stack_path": memmap_path,
        "filtered_rad_stack_shape": filtered_rad_stack.shape,
        "filtered_rad_stack_dtype": str(filtered_rad_stack.dtype),
        "start_tomo_ang": ome_range_deg[0][0],
        "end_tomo_ang": ome_range_deg[0][1],
        "tomo_num_imgs": tomo_num_imgs,
        "rot_axis_pos": rot_axis_pos,
        "pixel_size": pixel_size,
        "bad_frames": bad_frames,
        "recon_thresh": recon_thresh,
        "noise_obj_size": noise_obj_size,
        "min_hole_size": min_hole_size,
        "erosion_iter": erosion_iter,
        "dilation_iter": dilation_iter,
        "voxel_spacing": voxel_spacing,
        "x_center": x_center,
        "z_center": z_center,
        "x_length": x_length,
        "z_length": z_length,
    }

    tasks = list(enumerate(rows_to_recon))
    results = [None] * len(tasks)

    logging.info(f'Starting reconstruction for {len(tasks)} layers')

    with mp.Pool(
        processes=args.processes,
        initializer=init_reconstruct_worker,
        initargs=(worker_config,)
    ) as pool:
        for completed, (ii, tomo_mask) in enumerate(
            pool.imap_unordered(reconstruct_mask_for_layer, tasks), 1
        ):
            results[ii] = tomo_mask
            logging.info(f'Completed layer {completed}/{len(tasks)}')

    full_mask = np.array(results)

    gc.collect()
    os.remove(memmap_path)

    test_crds, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(
        v_bnds, voxel_spacing, x_center, z_center, x_length, z_length
    )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{output_stem}_tomo_mask.npz')

    np.savez_compressed(
        output_file,
        mask=full_mask.astype(bool),
        Xs=Xs,
        Ys=Ys,
        Zs=Zs,
        voxel_spacing=voxel_spacing
    )

    logging.info(f'Saved tomography mask to: {output_file}')


if __name__ == "__main__":
    mp.freeze_support()
    main()
