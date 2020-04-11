# coding=gbk

from __init__ import *

class Phenology:

    def __init__(self):
        self.this_class_arr = this_root + 'arr\\Phenology\\'
        self.this_class_tif = this_root + 'tif\\Phenology\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        ############# preprocess ##############
        # self.split_files()
        # self.data_transform_split_files()
        ############# preprocess ##############
        # self.hants()
        # self.SOS_EOS(0.4)
        # parts = [1,2]
        # thres = [0.1,0.2,0.4,0.5]
        # for part in parts:
        #     for threshold in thres:
        #         self.plot_SOS_EOS(part,threshold_i=threshold)

        # self.plot_SOS_bar()
        self.plot_part1_part2_bar()


        pass

    def split_files(self):
        fdir = this_root + 'data\\ndvi\\phenology\\clip\\'
        outdir = this_root + 'data\\ndvi\\phenology\\clip_yearly\\'
        Tools().mk_dir(outdir)
        years = np.array(range(1982, 2016))
        for y in tqdm(years):
            for f in os.listdir(fdir):
                year = int(f[:4])
                if y == year:
                    outdir_y = outdir + '{}\\'.format(y)
                    Tools().mk_dir(outdir_y)
                    shutil.copy(fdir + f, outdir_y + f)

    def data_transform_split_files(self):
        fdir = this_root + 'data\\ndvi\\phenology\\clip_yearly\\'
        outdir = this_root + 'data\\ndvi\\phenology\\per_pix_yearly\\'
        for year in os.listdir(fdir):
            print year
            fdir_i = fdir + year + '\\'
            outdir_i = outdir + year + '\\'
            Tools().mk_dir(outdir_i, force=1)
            self.data_transform(fdir_i, outdir_i)

        pass

    def data_transform(self,fdir, outdir):
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
                        array = np.array(array, dtype=float)
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
                void_dic['%03d.%03d' %(r, c)] = []
                void_dic_list.append('%03d.%03d' %(r, c))

        # print(len(void_dic))
        # exit()
        params = []
        for r in tqdm(range(row)):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%03d.%03d' %(r, c)].append(val)

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


    def __interp__(self,vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        x_new = np.linspace(min(inx),max(inx),365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new,y_new

    def hants(self):
        outdir = self.this_class_arr+'hants_smooth\\'
        fdir = this_root+'data\\ndvi\\phenology\\per_pix_yearly\\'
        for y in tqdm(os.listdir(fdir)):
            outdir_y = outdir+y+'\\'
            Tools().mk_dir(outdir_y,force=1)
            for f in os.listdir(fdir+y):
                dic = dict(np.load(fdir+y+'\\'+f).item())
                hants_dic = {}
                for pix in dic:
                    vals = dic[pix]
                    vals = np.array(vals)
                    std = np.std(vals)
                    if std == 0:
                        continue
                    xnew,ynew = self.__interp__(vals)
                    ynew = np.array([ynew])
                    # print np.std(ynew)
                    results=HANTS(sample_count=365,inputs=ynew,low=-10000, high=10000,
                    fit_error_tolerance=std)
                    result = results[0]
                    hants_dic[pix] = result
                np.save(outdir_y+f,hants_dic)


    def __search_left(self,vals,maxind,threshold_i):
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind-step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self,vals,maxind,threshold_i):
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind


    def SOS_EOS(self,threshold_i=0.2):
        out_dir = self.this_class_arr+'SOS_EOS\\threshold_{}\\'.format(threshold_i)
        Tools().mk_dir(out_dir,force=1)
        fdir = this_root+r'outdir_2020\arr\Phenology\hants_smooth\\'
        for y in tqdm(os.listdir(fdir)):
            year_dir = fdir+y+'\\'
            result_dic = {}
            for f in os.listdir(year_dir):
                dic = dict(np.load(year_dir+f).item())
                for pix in dic:
                    try:
                        vals = dic[pix]
                        maxind = np.argmax(vals)
                        start = self.__search_left(vals,maxind,threshold_i)
                        end = self.__search_right(vals,maxind,threshold_i)
                        result = [start,end,end-start]
                        result_dic[pix] = result
                        # print result
                    except:
                        pass
                        # plt.plot(vals)
                        # plt.show()
                        # exit()
                    # plt.plot(vals)
                    # plt.plot(range(start,end),vals[start:end],linewidth=4,zorder=99,color='r')
                    # plt.title('start:{} \nend:{} \nduration:{}'.format(start,end,end-start))
                    # plt.show()
            np.save(out_dir+y,result_dic)

    def plot_SOS_EOS(self,part,threshold_i=0.2):
        out_tif_dir = self.this_class_tif+'SOS_EOS\\'
        Tools().mk_dir(out_tif_dir)
        fdir = self.this_class_arr+'SOS_EOS\\threshold_{}\\'.format(threshold_i)

        year_range = []
        if part == 1:
            for y in range(1992,2003):
                year_range.append(str(y))
        elif part == 2:
            for y in range(2003,2016):
                year_range.append(str(y))
        else:
            raise IOError('error')


        start_mean = 0
        end_mean = 0
        duration_mean = 0
        flag = 0.
        for f in os.listdir(fdir):
            year = f.split('.')[0]
            if not year in year_range:
                continue
            flag += 1.
            dic = dict(np.load(fdir+f).item())

            start_dic = {}
            end_dic = {}
            duration_dic = {}
            for pix in dic:
                vals = dic[pix]
                start,end,duration = vals
                start_dic[pix] = start
                end_dic[pix] = end
                duration_dic[pix] = duration

            start_arr = DIC_and_TIF().pix_dic_to_spatial_arr(start_dic)
            end_arr = DIC_and_TIF().pix_dic_to_spatial_arr(end_dic)
            duration_arr = DIC_and_TIF().pix_dic_to_spatial_arr(duration_dic)

            start_mean += start_arr
            end_mean += end_arr
            duration_mean += duration_arr
        start_mean = start_mean/flag
        end_mean = end_mean/flag
        duration_mean = duration_mean/flag

        start_tif = out_tif_dir+'start_part{}_{}.tif'.format(part,threshold_i)
        end_tif = out_tif_dir+'end_part{}_{}.tif'.format(part,threshold_i)
        duration_tif = out_tif_dir+'duration_part{}_{}.tif'.format(part,threshold_i)

        DIC_and_TIF().arr_to_tif(start_mean,start_tif)
        DIC_and_TIF().arr_to_tif(end_mean,end_tif)
        DIC_and_TIF().arr_to_tif(duration_mean,duration_tif)

    def plot_SOS_bar(self):
        threshold = 0.1
        fdir = self.this_class_arr+r'SOS_EOS\threshold_{}\\'.format(threshold)

        start_mean_list = []
        end_mean_list = []
        for year in os.listdir(fdir):
            start_list = []
            end_list = []
            dic = dict(np.load(fdir+year).item())
            for pix in dic:
                start,end,duration = dic[pix]
                start_list.append(start)
                end_list.append(end)
            start_mean = np.mean(start_list)
            end_mean = np.mean(end_list)
            start_mean_list.append(start_mean)
            end_mean_list.append(end_mean)
        x = []
        y = []
        for i in range(len(start_mean_list)):
            x.append([start_mean_list[i],end_mean_list[i]])
            y.append([i+1982,i+1982])
            plt.plot([start_mean_list[i],end_mean_list[i]],[i+1982,i+1982],c='black',linewidth=4)
        plt.gca().invert_yaxis()
        # x = x[::-1]
        # # y = y[::-1]
        # for i in range(len(x)):
        #     plt.plot(x[i],y[i],c='black',linewidth=4)
        # plt.plot(start_mean_list)
        # plt.plot(end_mean_list)
        plt.title('threshold:{}'.format(threshold))
        plt.show()

    def plot_part1_part2_bar(self):
        fdir = self.this_class_tif+'SOS_EOS\\'
        start_end = ['start','end','duration']
        parts = ['part1','part2']
        threshold = 0.5
        x=[]
        y=[]
        for se in start_end:
            for part in parts:
                fname = '{}_{}_{}.tif'.format(se,part,threshold)
                arr = to_raster.raster2array(fdir+fname)[0]
                arr[arr<-9999]=np.nan
                mean = np.nanmean(arr)
                x.append('{}_{}_{}'.format(se,part,threshold))
                y.append(mean)
        plt.barh(x,y)
        plt.gca().invert_yaxis()
        plt.show()

        pass



def main():
    Phenology().run()
    pass


if __name__ == '__main__':
    main()