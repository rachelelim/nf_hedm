#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:04:28 2023

@author: lim37
"""

import numpy as np
from matplotlib import pyplot as plt
import os

from hexrd.transforms import xfcapi
from hexrd import rotations as rot
import nfutil
from hexrd.material import Material
from hexrd.valunits import valWUnit
import copy
import h5py

from cycler import cycler

color_cycle = [
    "#e31a10",
    "#336eef",

    "#f6d003",
    "#227910",
    "#6f269a",
    "#f6884e",
    "#52585a",
    "#f868ad",
    "#754916",
]

custom_cycler = cycler(
    color=color_cycle
)


plt.rcParams["axes.prop_cycle"] = custom_cycler


# %%

def generate_ori_map(grain_map, exp_maps, mat, id_remap=None):
    n_grains = len(exp_maps)

    rgb_image = np.zeros(
        [grain_map.shape[0], grain_map.shape[1], grain_map.shape[2], 3], dtype='float32')
    ori_map = np.zeros(
        [grain_map.shape[0], grain_map.shape[1], grain_map.shape[2], 4], dtype='float32')
    # rgb_image[:,:,:,3] = 1.

    for ii in np.arange(n_grains):
        if id_remap is not None:
            this_grain = np.where(np.squeeze(grain_map) == id_remap[ii])
        else:
            this_grain = np.where(np.squeeze(grain_map) == ii)
        if np.sum(this_grain[0]) > 0:

            ori = exp_maps[ii, :]

            rmats = xfcapi.makeRotMatOfExpMap(ori)
            # print(rmats.shape)
            quats = rot.quatOfRotMat(rmats.T).T
            # quat = np.vstack([quats[1:], quats[0]])

            rgb = mat.unitcell.color_orientations(
                rmats, ref_dir=np.array([0., 1., 0.]))

            ori_map[this_grain[0], this_grain[1],
                    this_grain[2], :] = quats

            # color mapping
            rgb_image[this_grain[0], this_grain[1],
                      this_grain[2], 0] = rgb[0][0]
            rgb_image[this_grain[0], this_grain[1],
                      this_grain[2], 1] = rgb[0][1]
            rgb_image[this_grain[0], this_grain[1],
                      this_grain[2], 2] = rgb[0][2]

    return rgb_image, ori_map


def convert_expmapfield_quats(expmaps_map):
    origin_dims = expmaps_map.shape
    xi0 = expmaps_map[:, :, :, 0].flatten()
    xi1 = expmaps_map[:, :, :, 1].flatten()
    xi2 = expmaps_map[:, :, :, 2].flatten()
    expmaps_list = np.stack([xi0, xi1, xi2])
    rmats = np.transpose(rot.rotMatOfExpMap(expmaps_list), axes=[0, 2, 1])
    # rgb = mat.unitcell.color_orientations(
    #             rmats, ref_dir=np.array([0., 1., 0.]))

    # print(rmats.shape)
    quats = rot.quatOfRotMat(rmats).T
    # dream3dquats = quats[:,[1,2,3,0]]

    q1 = np.reshape(quats[:, 1], [origin_dims[0],
                    origin_dims[1], origin_dims[2]])
    q2 = np.reshape(quats[:, 2], [origin_dims[0],
                    origin_dims[1], origin_dims[2]])
    q3 = np.reshape(quats[:, 3], [origin_dims[0],
                    origin_dims[1], origin_dims[2]])
    q4 = np.reshape(quats[:, 0], [origin_dims[0],
                    origin_dims[1], origin_dims[2]])
    dream3dquats = np.stack([q1, q2, q3, q4], axis=3)
    return dream3dquats  # , rgb


def write_to_h5(file_dir, file_name, data_array, data_name):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

    file_string = os.path.join(file_dir, file_name) + '.h5'
    hf = h5py.File(file_string, 'a')
    hf.require_dataset(data_name, np.shape(data_array), dtype=data_array.dtype)
    data = hf[data_name]
    data[...] = data_array
    hf.close()


def save_nf_data_for_paraview(file_dir, file_stem, grain_map, confidence_map, Xs, Ys, Zs,  mat, ori_vox_map=None, strain=None, tomo_mask=None, id_remap=None, voxNF_FF=None, avgvoxNF_FF=None, avgNF_FF=None, voxNF_avgNF=None):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

    print('Writing HDF5 data...')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(confidence_map, [1, 0, 2]), [2, 1, 0]), 'confidence')
    # write_to_h5(file_dir, file_stem + '_grain_map_data',
    #             np.transpose(np.transpose(misorientation_map, [1, 0, 2]), [2, 1, 0]), 'misorientation')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(grain_map, [1, 0, 2]), [2, 1, 0]), 'grain_map')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(Xs, [1, 0, 2]), [2, 1, 0]), 'Xs')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(Ys, [1, 0, 2]), [2, 1, 0]), 'Ys')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(Zs, [1, 0, 2]), [2, 1, 0]), 'Zs')

    if voxNF_FF is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(voxNF_FF, [1, 0, 2]), [2, 1, 0]), 'voxNF_FF')
    if avgvoxNF_FF is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(avgvoxNF_FF, [1, 0, 2]), [2, 1, 0]), 'avgvoxNF_FF')
    if avgNF_FF is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(avgNF_FF, [1, 0, 2]), [2, 1, 0]), 'avgNF_FF')
    if voxNF_avgNF is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(voxNF_avgNF, [1, 0, 2]), [2, 1, 0]), 'voxNF_avgNF')

    if strain is not None:
        print(len(strain))
        print(np.sum(grain_map == grain_map.max()))
        # for i in range(len(strain)):
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map-1, 0], [1, 0, 2]), [2, 1, 0]), 'strain_xx')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map-1, 1], [1, 0, 2]), [2, 1, 0]), 'strain_yy')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map-1, 2], [1, 0, 2]), [2, 1, 0]), 'strain_zz')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map-1, 3], [1, 0, 2]), [2, 1, 0]), 'strain_yz')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map-1, 4], [1, 0, 2]), [2, 1, 0]), 'strain_xz')
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(strain[grain_map-1, 5], [1, 0, 2]), [2, 1, 0]), 'strain_xy')

    if tomo_mask is not None:
        write_to_h5(file_dir, file_stem + '_grain_map_data',
                    np.transpose(np.transpose(tomo_mask, [1, 0, 2]), [2, 1, 0]), 'tomo_mask')
    # From unitcel the color is in hsl format
    # rgb_image, ori_map = generate_ori_map(grain_map, ori_list, mat, id_remap)
    # print(rgb_image.shape)
    # print(ori_map.shape)
    ori_map = ori_vox_map
    # print('Calculating IPF color')
    quats_map = convert_expmapfield_quats(ori_vox_map)
    # write_to_h5(file_dir, file_stem + '_grain_map_data',
    #             np.transpose(np.transpose(rgb_image, [1, 0, 2, 3]), [2, 1, 0, 3]), 'IPF_010')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(ori_map, [1, 0, 2, 3]), [2, 1, 0, 3]), 'expmaps')
    write_to_h5(file_dir, file_stem + '_grain_map_data',
                np.transpose(np.transpose(quats_map, [1, 0, 2, 3]), [2, 1, 0, 3]), 'quats')

    print('Writing XDMF...')
    xmdf_writer(file_dir, file_stem + '_grain_map_data')
    print('All done writing.')


def xmdf_writer(file_dir, file_name):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

    hf = h5py.File(os.path.join(file_dir, file_name)+'.h5', 'r')
    k = list(hf.keys())
    totalsets = len(k)
    # What are the datatypes
    datatypes = np.empty([totalsets], dtype=object)
    databytes = np.zeros([totalsets], dtype=np.int8)
    # note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))
    dims = np.ones([totalsets, 4], dtype=int)
    for i in np.arange(totalsets):
        datatypes[i] = hf[k[i]].dtype
        databytes[i] = hf[k[i]].dtype.itemsize
        s = np.shape(hf[k[i]])
        if len(s) > 4:
            print(
                'An array has greater than 4 dimensions - this writer cannot handle that')
            hf.close()
            # return
        dims[i, 0:len(s)] = s

    hf.close()

    filename = os.path.join(file_dir, file_name) + '.xdmf'
    f = open(filename, 'w')

    # Header for xml file
    f.write('<?xml version="1.0"?>\n')
    f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"[]>\n')
    f.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
    f.write(' <Domain>\n')
    f.write('\n')
    f.write('  <Grid Name="Cell Data" GridType="Uniform">\n')
    f.write('    <Topology TopologyType="3DCoRectMesh" Dimensions="' +
            str(dims[0, 0]+1) + ' ' + str(dims[0, 1]+1) + ' ' + str(dims[0, 2]+1) + '"></Topology>\n')
    f.write('    <Geometry Type="ORIGIN_DXDYDZ">\n')
    f.write('      <!-- Origin -->\n')
    f.write('      <DataItem Format="XML" Dimensions="3">0 0 0</DataItem>\n')
    f.write('      <!-- DxDyDz (Spacing/Resolution)-->\n')
    f.write('      <DataItem Format="XML" Dimensions="3">1 1 1</DataItem>\n')
    f.write('    </Geometry>\n')
    f.write('\n')

    for i in np.arange(totalsets):
        if dims[i, 3] == 1:
            f.write('      <Attribute Name="' +
                    k[i] + '" AttributeType="Scalar" Center="Cell">\n')
            f.write('      <DataItem Format="HDF" Dimensions="' + str(dims[i, 0]) + ' ' + str(dims[i, 1]) + ' ' + str(
                dims[i, 2]) + ' ' + str(dims[i, 3]) + '" NumberType="' + str(datatypes[i]) + '" Precision="' + str(databytes[i]) + '" >\n')
            f.write('       ' + file_name + '.h5:/' + k[i] + '\n')
            f.write('      </DataItem>\n')
            f.write('       </Attribute>\n')
            f.write('\n')
        elif dims[i, 3] > 1:
            f.write('      <Attribute Name="' +
                    k[i] + '" AttributeType="Vector" Center="Cell">\n')
            f.write('      <DataItem Format="HDF" Dimensions="' + str(dims[i, 0]) + ' ' + str(dims[i, 1]) + ' ' + str(
                dims[i, 2]) + ' ' + str(dims[i, 3]) + '" NumberType="' + str(datatypes[i]) + '" Precision="' + str(databytes[i]) + '" >\n')
            f.write('       ' + file_name + '.h5:/' + k[i] + '\n')
            f.write('      </DataItem>\n')
            f.write('       </Attribute>\n')
            f.write('\n')
        else:
            print('Wrong array size of ' +
                  str(dims[i, :]) + ' in array ' + str(i))

    # End the xmf file
    f.write('  </Grid>\n')
    f.write(' </Domain>\n')
    f.write('</Xdmf>\n')

    f.close()


def stitch_nf_diffraction_volumes(output_dir, output_stem, paths, raw_paths, FF_path, offsets, tomo_mask=None, ori_tol=0.05, overlap=0, write=True):
    '''
    This function stiches multiple NF diffraction volumes:
    Inputs:
        paths: .npz file locations
            size: length number of diffraction volumes
        offsets: separation of diffraction volumes
            size: length number of diffraction volumes
            These are the motor positions, thus they are likely inverted such that offset[0] will be the,
            top most diffraction volume; however, the actual value will be the smallest motor position
        ori_tol: orientation tolerance to merge adjacet grains
            size: single valued, in degrees
        overlap: overlap of diffraction volumes
            size: single valued, in voxels along stacking direction
            example: if you have 10 micron overlap, and your voxel size is 5 micron, overlap=2
    Assumptions:
        - stacking direction will be the shortest of the three normal directions
        - merging of overlap can be done by a simple confidence check
        - grain maps have the same dimensions
    '''
    print('Loading data.')
    # Some data lists initialization
    grain_map_list = []
    conf_map_list = []
    Xs_list = []
    Ys_list = []
    Zs_list = []
    nf_to_ff_id_map = []
    exp_maps_list = []
    voxNF_FF_map_list = []
    mask_list = []

    # Load data into lists
    for i, p in enumerate(paths):
        nf_recon = np.load(p)
        grain_map_list.append(np.flip(nf_recon['grain_map'], 0))
        conf_map_list.append(np.flip(nf_recon['confidence_map'], 0))
        # exp_map_list.append(nf_recon['ori_list'])
        Xs_list.append(np.flip(nf_recon['Xs'], 0))
        Ys_list.append(np.flip(nf_recon['Ys'], 0) - offsets[i])
        Zs_list.append(np.flip(nf_recon['Zs'], 0))
        nf_to_ff_id_map.append(nf_recon['id_remap'])
        if tomo_mask is not None:
            mask_list.append(nf_recon['tomo_mask'])

    dims = np.shape(grain_map_list[0])

    FF_data = np.loadtxt(FF_path)
    FF_expmaps = FF_data[:, 3:6]

    syms = rot.quatOfLaueGroup('oh').T

    # mask = np.ones(dims).astype('bool')  # raw_data['mask']
    for i, p in enumerate(raw_paths):
        print(p)
        raw_data = np.load(p)
        raw_idx = raw_data['raw_idx']
        raw_misorientation = raw_data['raw_misorientation']
        raw_exp_maps = raw_data['raw_exp_maps']

        grain_ids = np.unique(raw_idx).astype(int)

        for count, gid in enumerate(grain_ids):
            grain_voxels = raw_idx == gid
            # voxelized grain expmaps
            grain_expmaps = raw_exp_maps[grain_voxels]
            this_grain_quats = rot.quatOfExpMap(
                grain_expmaps.T)  # voxelized grain quats
            grain_expmap_avg = np.average(grain_expmaps, axis=0)
        print(i)
        mask = mask_list[i]

        expmaps_map = np.zeros([dims[0], dims[1], dims[1], 3])
        expmaps_map[mask] = raw_exp_maps
        voxNF_FF_map = np.zeros(dims)
        voxNF_FF_map[mask] = raw_misorientation
        exp_maps_list.append(np.flip(expmaps_map, 0))
        voxNF_FF_map_list.append(np.flip(voxNF_FF_map, 0))

        # masks.append(nf_recon['tomo_mask'])
    # What are the dimensions of our arrays?
    # Since numpy dimensions are strange this is Y X Z

    # Initialize the merged data arrays
    num_vols = i + 1
    grain_map_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2]), dtype=np.int32)
    confidence_map_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2]), dtype=np.float32)
    Xs_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                       dims[1], dims[2]), dtype=np.float32)
    Ys_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                       dims[1], dims[2]), dtype=np.float32)
    Zs_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                       dims[1], dims[2]), dtype=np.float32)
    if tomo_mask is not None:
        mask_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                              dims[1], dims[2]), dtype=np.float32)
    vertical_position_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2]), dtype=np.float32)
    exp_maps_full = np.zeros(
        ((dims[0]-overlap*2)*num_vols, dims[1], dims[2], 3), dtype=np.float32)
    voxNF_FF_map_full = np.zeros(((dims[0]-overlap*2)*num_vols,
                                  dims[1], dims[2]), dtype=np.float32)

    print('Data Loaded.')
    print('Merging diffraction volumes.')
    # Run the merge
    if overlap != 0:
        # There is an overlap, use confidence to define which voxel to pull at each overlap
        # First assume there is no overlap
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        print(dims)
        for vol in np.arange(num_vols):
            grain_map_full[start:stop, :,
                           :] = grain_map_list[vol][overlap:dims[0]-overlap, :, :]
            confidence_map_full[start:stop, :,
                                :] = conf_map_list[vol][overlap:dims[0]-overlap, :, :]
            exp_maps_full[start:stop, :,
                          :, :] = exp_maps_list[vol][overlap:dims[0]-overlap, :, :, :]
            Xs_full[start:stop, :, :] = Xs_list[vol][overlap:dims[0]-overlap, :, :]
            Ys_full[start:stop, :, :] = Ys_list[vol][overlap:dims[0]-overlap, :, :]
            Zs_full[start:stop, :, :] = Zs_list[vol][overlap:dims[0]-overlap, :, :]
            if tomo_mask is not None:
                mask_full[start:stop, :,
                          :] = mask_list[vol][overlap:dims[0]-overlap, :, :]
            voxNF_FF_map_full[start:stop, :,
                              :] = voxNF_FF_map_list[vol][overlap:dims[0]-overlap, :, :]
            vertical_position_full[start:stop, :, :] = offsets[vol]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2
            print(Ys_list[vol][overlap:dims[0]-overlap, :, :].min())
            print(Ys_list[vol][overlap:dims[0]-overlap, :, :].max())
        # Now handle the overlap regions
        # Where are they?  They are the overlap voxels on either side of the volume division,
        # where the division-overlap region has to be checked against the first overlap voxels
        # of the next region, and a similar number of voxels into that region need to be checked
        # against the final voxels of the first
        # We have number of diffraction volumes - 1 overlap chunks
        # This is actually the first index of the next diffraction volume
        division_line = dims[0] - overlap*2
        for overlap_region in np.arange(num_vols-1):
            # Handle one side
            # This will be true at voxels which need replacing
            replace_voxels = np.signbit(
                confidence_map_full[division_line-overlap:division_line, :, :]-conf_map_list[overlap_region+1][0:overlap, :, :])
            grain_map_full[division_line-overlap:division_line, :,
                           :][replace_voxels] = grain_map_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            confidence_map_full[division_line-overlap:division_line, :,
                                :][replace_voxels] = conf_map_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            exp_maps_full[division_line-overlap:division_line, :,
                          :, :][replace_voxels] = exp_maps_list[overlap_region+1][0:overlap, :, :, :][replace_voxels]
            Xs_full[division_line-overlap:division_line, :,
                    :][replace_voxels] = Xs_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            Ys_full[division_line-overlap:division_line, :,
                    :][replace_voxels] = Ys_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            Zs_full[division_line-overlap:division_line, :,
                    :][replace_voxels] = Zs_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            if tomo_mask is not None:
                mask_full[division_line-overlap:division_line, :,
                          :][replace_voxels] = mask_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            voxNF_FF_map_full[division_line-overlap:division_line, :,
                              :][replace_voxels] = voxNF_FF_map_list[overlap_region+1][0:overlap, :, :][replace_voxels]
            vertical_position_full[division_line-overlap:division_line,
                                   :, :][replace_voxels] = offsets[overlap_region+1]
            division_line = division_line + dims[0] - overlap*2
    else:
        # There is no overlap, scans will be stacked directly
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop, :,
                           :] = grain_map_list[vol][overlap:dims[0]-overlap, :, :]
            confidence_map_full[start:stop, :,
                                :] = conf_map_list[vol][overlap:dims[0]-overlap, :, :]
            exp_maps_full[start:stop, :,
                          :, :] = exp_maps_list[vol][overlap:dims[0]-overlap, :, :, :]
            Xs_full[start:stop, :, :] = Xs_list[vol][overlap:dims[0]-overlap, :, :]
            Ys_full[start:stop, :, :] = Ys_list[vol][overlap:dims[0]-overlap, :, :]
            Zs_full[start:stop, :, :] = Zs_list[vol][overlap:dims[0]-overlap, :, :]
            if tomo_mask is not None:
                mask_full[start:stop, :,
                          :] = mask_list[vol][overlap:dims[0]-overlap, :, :]
            voxNF_FF_map_full[start:stop, :,
                              :] = voxNF_FF_map_list[vol][overlap:dims[0]-overlap, :, :]
            vertical_position_full[start:stop, :, :] = offsets[vol]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2
    # if write is True:
        # save_nf_data_for_paraview(output_dir, output_stem, grain_map_full,
        #                           confidence_map_full, Xs_full, Ys_full, Zs_full, exp_map_list, material)  # , strain)

    gids = np.unique(grain_map_full)
    avg_NF_expmap = np.zeros([len(FF_data), 3])
    avg_NF_quat = np.zeros([len(FF_data), 4])
    avgNF_FF_map_full = np.zeros(voxNF_FF_map_full.shape)
    vox_NF_avgNF_map_full = np.zeros(voxNF_FF_map_full.shape)
    avgvoxNF_FF_map_full = np.zeros(voxNF_FF_map_full.shape)

    grain_map_full[confidence_map_full < 0.45] = -1
    gids = np.unique(grain_map_full)

    for count, gid in enumerate(gids):
        # print(gid)
        grain_voxels = grain_map_full == gid
        # print('nvoxels = ' + str(np.sum(grain_voxels)))
        # voxelized grain expmaps
        grain_expmaps = exp_maps_full[grain_voxels]
        this_grain_quats = rot.quatOfExpMap(
            grain_expmaps.T)  # voxelized grain quats
        grain_expmap_avg = np.average(grain_expmaps, axis=0)
        avg_NF_expmap[gid, :] = grain_expmap_avg
        avg_NF_quat[gid, :] = rot.quatOfExpMap(grain_expmap_avg)

        # between avg NF and FF
        mis = rot.misorientation(np.atleast_2d(rot.quatOfExpMap(
            grain_expmap_avg)).T, np.atleast_2d(rot.quatOfExpMap(FF_expmaps[gid])).T, (syms.T,))
        avgNF_FF_map_full[grain_voxels] = np.degrees(mis[0])
        avgvoxNF_FF_map_full[grain_voxels] = np.average(
            voxNF_FF_map_full[grain_voxels])

        # between voxelized NF and avg NF
        if np.sum(grain_voxels) == 1:
            print('Grain %d has 1 voxel' % gid)
            this_mis_NF = rot.misorientation(np.atleast_2d(rot.quatOfExpMap(
                grain_expmap_avg)).T, np.atleast_2d(this_grain_quats).T, (syms.T,))
        elif np.sum(grain_voxels) > 1:
            if np.sum(grain_voxels) < 10:
                print('Grain %d has %d voxels' % (gid, np.sum(grain_voxels)))
            this_mis_NF = rot.misorientation(np.atleast_2d(
                rot.quatOfExpMap(grain_expmap_avg)).T, this_grain_quats, (syms.T,))
        vox_NF_avgNF_map_full[grain_voxels] = np.degrees(this_mis_NF[0])

    grain_map_full += 1
    grain_map_full[confidence_map_full < 0.45] = 0
    print('Diffraction volumes merged.')
    if tomo_mask is not None:
        return grain_map_full, confidence_map_full, Xs_full, Ys_full, Zs_full, exp_maps_full, avgNF_FF_map_full, voxNF_FF_map_full, vox_NF_avgNF_map_full, avgvoxNF_FF_map_full, mask_full

    else:
        return grain_map_full, confidence_map_full, Xs_full, Ys_full, Zs_full, exp_maps_full, avgNF_FF_map_full, voxNF_FF_map_full, vox_NF_avgNF_map_full, avgvoxNF_FF_map_full, mask_full
# %%


state = 'init'
nlayers = 4
sample = 'V-TT-1'

mat_name = 'V'
max_tth = None  # degrees, if None is input max tth will be set by the geometry
mat_file = '../../materials.h5'
beam_energy = 51.998
# mask = np.ones(vol_dims).astype('bool') #raw_data['mask']

tomo_mask_fle = '/usr/workspace/lim37/CHESS_2025-03/V/V-TT-1/tomo/V-TT-1_init_tomo_clean.npz'


mat = Material(name=mat_name, material_file=mat_file, dmin=valWUnit(
    'lp', 'length',  0.05, 'nm'), kev=valWUnit('kev', 'energy', beam_energy, 'keV'))


fdir = '/usr/workspace/lim37/CHESS_2025-03/V/%s/NF/' % sample
fstem = '/usr/workspace/lim37/CHESS_2025-03/V/%s/NF/%s_S0_L%%d_grain_map_data.npz' % (
    sample, sample)
grains_file = 'V-TT-1_init_dream3d_Nov2025_S000_master.out'
raw_NF_stem = '/usr/workspace/lim37/CHESS_2025-03/V/%s/NF/%s_S0_L%%d_raw_reconstruction.npz' % (
    sample, sample)


layer_height = 0.3
layer_center_offset = np.array([0.4, 0.2, 0.0, -0.2])
overlap = 0.0

vox_size = 0.005
cross_section_dim = 1.5
height = nlayers*layer_height - (nlayers-1)*overlap

vox_width = int(cross_section_dim/vox_size)
vox_height = int(height/vox_size)
vox_per_layer = int(layer_height/vox_size)
overlap_vox = int(overlap/vox_size)
# %%

vol_dims = (vox_per_layer, int(cross_section_dim/vox_size),
            int(cross_section_dim/vox_size))

fnames = []
raw_fnames = []
for i in range(nlayers):
    fnames.append(fstem % (i))
    raw_fnames.append(raw_NF_stem % (i))
# fnames = [fstem%(state, 2, state)]

grain_map_full, confidence_map_full, Xs_full, Ys_full, Zs_full, \
    expmaps_map, avgNF_FF_map_full, voxNF_FF_map_full, vox_NF_avgNF_map_full, \
    avgvoxNF_FF_map_full, mask = stitch_nf_diffraction_volumes(fdir, 'NF_stitched',
                                                               fnames, raw_fnames,
                                                               grains_file, layer_center_offset,
                                                               ori_tol=0.05, overlap=overlap_vox, tomo_mask=1)

# %%
grain_data = np.loadtxt(grains_file)
print(grains_file)
strain = grain_data[:, -6:]

good_grains = grain_data[:, 2] < 5e-3
yloc = grain_data[good_grains, 7]
hstatic = np.sum(strain[:, :3], axis=1)

plt.plot(yloc, hstatic[good_grains], '.')

# poly = np.polyfit(yloc, hstatic[good_grains], 1)

# plt.plot(yloc, poly[0]*yloc+poly[1])
# %%
strainxx = strain[:, 0]  # -(poly[0]*grain_data[:, 7]+poly[1])/3
strainyy = strain[:, 1]  # -(poly[0]*grain_data[:, 7]+poly[1])/3
strainzz = strain[:, 2]  # -(poly[0]*grain_data[:, 7]+poly[1])/3
# %%
plt.plot(yloc, hstatic[good_grains], '.')
plt.plot(grain_data[:, 7], strainxx, '.')
plt.plot(grain_data[:, 7], strainyy, '.')
plt.plot(grain_data[:, 7], strainzz, '.')

strain[:, 0] = strainxx
strain[:, 1] = strainyy
strain[:, 2] = strainzz

# %%

nfutil.save_nf_data(fdir, '%s_dream3d_4layer' % state, grain_map_full, confidence_map_full,
                    Xs_full, Ys_full, Zs_full, expmaps_map, tomo_mask=mask)
# %%
save_nf_data_for_paraview(fdir, '%s_dream3d_4layer' % state, grain_map_full,
                          confidence_map_full, Xs_full, Ys_full, Zs_full, mat,
                          ori_vox_map=expmaps_map, strain=strain,
                          voxNF_FF=voxNF_FF_map_full, avgvoxNF_FF=avgvoxNF_FF_map_full,
                          avgNF_FF=avgNF_FF_map_full, voxNF_avgNF=vox_NF_avgNF_map_full, tomo_mask=mask)

# %%
