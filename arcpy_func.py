# coding=gbk

import os
import arcpy
from arcpy.sa import *

this_root = 'D:\\project_phenology\\'


def mk_dir( dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)


def resample(in_dir,out_dir):
    # ndvi_8km_dir = this_root+'CCI\\0.25\\tif\\'
    # ndvi_0_5_dir = this_root+'CCI\\0.5\\tif\\'
    mk_dir(out_dir)
    for f in os.listdir(in_dir):
        if f.endswith('.tif'):
            print(f)
            arcpy.Resample_management(in_dir+f,out_dir+f,"0.25","NEAREST")



def define_swe_projection():
    fdir = r'D:\project05\GLOBSWE\tif\SWE_avg\\'
    fdir = r'D:\project05\GLOBSWE\tif\SWE_max\\'
    for f in os.listdir(fdir):
        print f
        sr = arcpy.SpatialReference("North Pole Lambert Azimuthal Equal Area")
        arcpy.DefineProjection_management(fdir+f, sr)

def re_projection_swe():
    # f = r'D:\project05\GLOBSWE\tif\SWE_avg\198202.tif'
    # outf = r'D:\project05\GLOBSWE\test123.tif'
    # prj = r'D:\project05\MXD'
    fdir = r'D:\project05\GLOBSWE\tif\SWE_max\\'
    outdir = r'D:\project05\GLOBSWE\re_proj\SWE_max\\'
    for f in os.listdir(fdir):
        if f.endswith('.tif'):
            print f
            in_f = fdir+f
            outf = outdir+f
            arcpy.ProjectRaster_management(in_f, outf,
                                           r"C:\Program Files (x86)\ArcGIS\Desktop10.2\Reference Systems\georef1.prj", "BILINEAR", "0.5",
                                           "#", "#", "#")
    pass


def mapping(current_dir,tif,outjpeg,title,mxd_file):

    mxd = arcpy.mapping.MapDocument(mxd_file)
    df0 = arcpy.mapping.ListDataFrames(mxd)[0]

    workplace = "RASTER_WORKSPACE"

    lyr = arcpy.mapping.ListLayers(mxd, 'tif', df0)[0]
    lyr.replaceDataSource(current_dir,workplace,tif)

    for textElement in arcpy.mapping.ListLayoutElements(mxd, "TEXT_ELEMENT"):
        if textElement.name == 'title':
            textElement.text = (title)

    arcpy.mapping.ExportToJPEG(mxd,outjpeg,data_frame='PAGE_LAYOUT',df_export_width=mxd.pageSize.width,df_export_height=mxd.pageSize.height,color_mode='24-BIT_TRUE_COLOR',resolution=300,jpeg_quality=100)


def do_mapping_recovery_time():


    mode = {'pick_non_growing_season_events':'None Growing Season',
            'pick_pre_growing_season_events':'Early Growing Season',
            'pick_post_growing_season_events':'Late Growing Season'
    }
    window_start = [6]
    # window_end = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    window_end = [12]
    for ws in window_start:
        for we in window_end:
            for m in mode:
                current_dir = r'D:\project05\tif\recovery_time\{}_plot_gen_recovery_time_{}_{}\\'.format(m,ws,we)
                tif = '{}.tif'.format(m)
                title = mode[m]+' {}-{}'.format(ws,we)
                outjpeg = r'D:\project05\png\recovery_time\composite\\{}.jpg'.format(title)

                mxd_file = r'D:\project05\MXD\recovery_time2.mxd'
                print title
                mapping(current_dir,tif,outjpeg,title,mxd_file)

    pass



def do_mapping_recovery_time1():


    mode = {'pick_non_growing_season_events':'None Growing Season',
            'pick_pre_growing_season_events':'Early Growing Season',
            'pick_post_growing_season_events':'Late Growing Season'
    }
    for m in mode:
        current_dir = r'D:\project05\tif\recovery_time\{}_plot_gen_recovery_time\\'.format(m)
        tif = 'global_mask.tif'
        title = mode[m]
        outjpeg = r'D:\project05\png\recovery_time\recovery\{}.jpg'.format(title)
        mxd_file = r'D:\project05\MXD\recovery_time2.mxd'
        print title
        mapping(current_dir,tif,outjpeg,title,mxd_file)

    current_dir = r'D:\project05\tif\recovery_time\\'
    tif = 'recovery_time_mix.tif'
    outjpeg = r'D:\project05\png\recovery_time\recovery\Mix_Recovery_time.jpg'
    title = 'Mix Recovery time'
    mxd_file = r'D:\project05\MXD\recovery_time2.mxd'
    print title
    mapping(current_dir, tif, outjpeg, title, mxd_file)
    pass


def do_mapping_recovery_time2():


    mode = {'pick_non_growing_season_events':'None Growing Season',
            'pick_pre_growing_season_events':'Early Growing Season',
            'pick_post_growing_season_events':'Late Growing Season'
    }
    for mark in ['in','out']:
        for m in mode:
            current_dir = r'D:\project05\tif\recovery_time\in_or_out\\'
            tif = '{}_{}'.format(m,mark)
            if mark == 'in':
                title = 'Drought in '+mode[m]+' and Recovered IN Current Growing Season'
            elif mark == 'out':
                title = 'Drought in '+mode[m]+' and Recovered not IN Current Growing Season'
            else:
                raise IOError()
            outjpeg = r'D:\project05\png\recovery_time\in_out\{}.jpg'.format(title)
            mxd_file = r'D:\project05\MXD\recovery_time2.mxd'
            print title
            mapping(current_dir,tif,outjpeg,title,mxd_file)



def do_mapping_recovery_time3():


    current_dir = r'D:\project05\new_2020\tif\recovery_time_in_out\\'
    # tif = r'early_in_arr.tif'
    # title = 'Early IN'

    # tif = r'early_out_arr.tif'
    # title = 'Early OUT'
    #
    # tif = r'late_in_arr.tif'
    # title = 'Late IN'
    #
    tif = r'late_out_arr.tif'
    title = 'Late OUT'


    outjpeg = r'D:\project05\new_2020\jpg\in_out\{}.jpg'.format(title)
    mxd_file = r'D:\project05\MXD\recovery_time2.mxd'
    print title
    mapping(current_dir,tif,outjpeg,title,mxd_file)



def do_resample_HWSD():
    f = r'D:\project05\branch2020\tif\HWSD\T_SAND.tif'
    outf = r'D:\project05\branch2020\tif\HWSD\T_SAND_resample.tif'
    arcpy.Resample_management(f,outf,"0.5","NEAREST")

    pass


def main():

    indir = r'D:\project_phenology\data\NDVI\tif_8km_bi_weekly\\'
    outdir = r'D:\project_phenology\data\NDVI\tif_0.25_bi_weekly\\'
    resample(indir,outdir)
    # do_mapping_recovery_time3()
    # do_resample_HWSD()
    pass
if __name__ == '__main__':
    main()