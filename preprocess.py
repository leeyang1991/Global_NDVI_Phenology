# coding=gbk

from __init__ import *


def hdf_to_tif():
    out_dir = this_root+'data\\NDVI\\tif_8km_bi_weekly\\'
    Tools().mk_dir(out_dir)
    fdir = this_root+'data\\NDVI\\GIMMS_NDVI\\'
    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.hdf'):
            continue
        # if not f == 'ndvi3g_geo_v1_2014_0106.hdf':
        #     continue
        # print(f)
        year = f.split('.')[0].split('_')[-2]
        # print(year)
        hdf = h5py.File(fdir+f, 'r')
        for i in range(len(hdf['time'])):
            arr = hdf['ndvi'][i]
            lon = hdf['lon']
            lat = hdf['lat']
            time = hdf['time'][i]
            time_str = str(time)
            if time_str.endswith('.5'):
                date = year+'%02d'%int(time)+'15'
            else:
                date = year+'%02d'%int(time)+'01'

            newRasterfn = out_dir+'{}.tif'.format(date)
            longitude_start = lon[0]
            latitude_start = lat[0]
            pixelWidth = lon[1]-lon[0]
            pixelHeight = lat[1]-lat[0]
            arr = np.array(arr,dtype=float)
            # print(arr.dtype)
            grid = arr > - 10000
            arr[np.logical_not(grid)] = -999999
            # import time
            # time.sleep(1)
            # plt.imshow(arr)
            # plt.show()
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
            pass



def data_transform(fdir, outdir):
    # 不可并行，内存不足
    Tools().mk_dir(outdir)
    # 将空间图转换为数组
    # per_pix_data
    flist = os.listdir(fdir)
    date_list = []
    for f in flist:
        if f.endswith('.tif'):
            date = f.split('.')[0]
            date_list.append(date)
    date_list.sort()
    all_array = []
    for d in tqdm(date_list, 'loading...'):
        # for d in date_list:
        for f in flist:
            if f.endswith('.tif'):
                if f.split('.')[0] == d:
                    # print(d)
                    array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                    array = np.array(array,dtype=np.int16)
                    # print np.min(array)
                    # print type(array)
                    # plt.imshow(array)
                    # plt.show()
                    all_array.append(array)

    row = len(all_array[0])
    col = len(all_array[0][0])

    void_dic = {}
    void_dic_list = []
    for r in range(row):
        for c in range(col):
            void_dic[(r, c)] = []
            void_dic_list.append((r, c))

    # print(len(void_dic))
    # exit()
    params = []
    for r in tqdm(range(row)):
        for c in range(col):
            for arr in all_array:
                val = arr[r][c]
                void_dic[(r, c)].append(val)

    # for i in void_dic_list:
    #     print(i)
    # exit()
    flag = 0
    temp_dic = {}
    for key in tqdm(void_dic_list, 'saving...'):
        flag += 1
        # print('saving ',flag,'/',len(void_dic)/100000)
        temp_dic[key] = void_dic[key]
        if flag % 10000 == 0:
            # print('\nsaving %02d' % (flag / 10000)+'\n')
            np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
            temp_dic = {}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)



def split_files():

    fdir = this_root+'data\\NDVI\\tif_0.25_bi_weekly\\'
    outdir = this_root+'data\\NDVI\\tif_0.25_bi_weekly_yearly\\'
    Tools().mk_dir(outdir)
    years = np.array(range(1982,2016))
    for y in tqdm(years):
        for f in os.listdir(fdir):
            year = int(f[:4])
            if y == year:
                outdir_y = outdir+'{}\\'.format(y)
                Tools().mk_dir(outdir_y)
                shutil.copy(fdir+f,outdir_y+f)


def data_transform_split_files():
    fdir = this_root+'data\\NDVI\\tif_0.25_bi_weekly_yearly\\'
    outdir = this_root+'data\\NDVI\\per_pix_0.25_bi_weekly_yearly\\'
    for year in os.listdir(fdir):
        print year
        fdir_i = fdir+year+'\\'
        outdir_i = outdir+year+'\\'
        Tools().mk_dir(outdir_i,force=1)
        data_transform(fdir_i,outdir_i)

    pass


def main():
    # hdf_to_tif()
    split_files()
    data_transform_split_files()
    # fdir = this_root+'data\\NDVI\\tif_0.5_bi_weekly\\'
    # outdir = this_root+'data\\NDVI\\per_pix_tif_0.5_bi_weekly\\'
    # data_transform(fdir,outdir)

    pass



if __name__ == '__main__':
    main()