# coding=gbk

from analysis import *
from HANTS import *

def sleep(t=1):
    time.sleep(t)


class Phenology_old:

    def __init__(self):
        self.this_class_arr = this_root + 'outdir_2020\\arr\\Phenology\\'
        self.this_class_tif = this_root + 'outdir_2020\\tif\\Phenology\\'
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
        # self.plot_part1_part2_bar()
        # self.slope()
        # self.slope_s1_s2()
        # self.check_slope()
        # self.SOS_EOS_S1_S2_S3()
        # self.s1_s2_s3_to_mon()
        # self.start_end_part1_part2_change()
        # self.s1_s2_s3_to_tif()
        # self.s1_s2_s3_to_tif_part12()
        # self.s1_s2_s3_anomaly()
        # self.SOS_EOS_Duration_to_tif()
        self.SOS_EOS_Duration_anomaly()

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



    def __search_left_slpoe(self,vals,maxind):
        left_point_ind = 999999
        left_point_val = np.nan
        selected_vals = vals[:maxind]
        for step in range(len(selected_vals)):
            ind = maxind - step
            if ind-1 < 0:
                break
            val = vals[ind]
            val_left = vals[ind-1]
            val_right = vals[ind+1]
            if val < val_left and val > val_right:
                left_point_val = val
                left_point_ind = ind
                break
        if left_point_ind > 99999:
            raise IOError('search failed')
        return left_point_ind,left_point_val


    def __search_right_slpoe(self,vals,maxind):
        right_point_ind = 999999
        right_point_val = np.nan
        selected_vals = vals[maxind:]
        # exit()
        for step in range(len(selected_vals)):
            ind = maxind + step
            if ind+1 >= len(vals):
                # plt.plot(vals)
                # plt.show()
                raise IOError('exceed')
            val = vals[ind]
            val_left = vals[ind-1]
            val_right = vals[ind+1]
            if val > val_left and val < val_right:
                right_point_val = val
                right_point_ind = ind
                break
        if right_point_ind > 99999:
            raise IOError('search failed')
        # plt.plot(vals)
        # val_str = []
        # for i in vals:
        #     val_str.append(str(i))
        # for i in range(len(vals)):
        #     plt.text(i,vals[i],val_str[i])
        # plt.plot(np.array(range(len(selected_vals)))+maxind,selected_vals,linewidth=4,zorder=9)
        # plt.scatter(right_point_ind,right_point_val,zorder=99)
        # plt.show()
        return right_point_ind,right_point_val


    def slope(self):
        outdir = self.this_class_arr + 'slope\\slope\\'
        Tools().mk_dir(outdir)
        fdir = this_root + r'outdir_2020\arr\Phenology\hants_smooth\\'
        for y in tqdm(os.listdir(fdir)):
            one_year_dic = {}
            for f in os.listdir(fdir + y):
                dic = dict(np.load(fdir + y + '\\' + f).item())
                one_year_dic.update(dic)
            slope_y_dic = {}
            for pix in one_year_dic:
                vals = one_year_dic[pix]
                slopes = []
                for i in range(len(vals)):
                    if i - 1 < 0:
                        slope_i = vals[1] - vals[0]
                    else:
                        slope_i = vals[i] - vals[i - 1]
                    slopes.append(slope_i)
                slope_y_dic[pix] = slopes
                # print slopes
            # slope_dic[y] = slope_y_dic
            np.save(outdir + y, slope_y_dic)

        pass


    def slope_s1_s2(self):
        outdir = self.this_class_arr+'slope\\'
        Tools().mk_dir(outdir)
        fdir = this_root+r'outdir_2020\arr\Phenology\hants_smooth\\'
        slope_dic = {}
        for y in os.listdir(fdir):
            one_year_dic = {}
            for f in os.listdir(fdir+y):
                dic = dict(np.load(fdir+y+'\\'+f).item())
                one_year_dic.update(dic)
            s1_s2_dic = {}
            for pix in tqdm(one_year_dic,desc=y):
                vals = one_year_dic[pix]
                slopes = []
                for i in range(len(vals)):
                    if i-1 < 0:
                        slope_i = vals[1]-vals[0]
                    else:
                        slope_i = vals[i]-vals[i-1]
                    slopes.append(slope_i)
                max_ind = int(np.argmax(slopes))
                right_val = slopes[max_ind:]  # take right values
                min_ind = int(np.argmin(right_val)) + max_ind
                # left_point_ind,left_point_val = self.__search_left_slpoe(slopes,max_ind)
                s1_s2_point = (max_ind, min_ind)
                s1_s2_dic[pix] = s1_s2_point
                # try:
                #     right_point_ind,right_point_val = self.__search_right_slpoe(slopes,max_ind)
                #     s1_s2_point = (max_ind,right_point_ind)
                #     s1_s2_dic[pix] = s1_s2_point
                # except:
                #     pass
            slope_dic[y]=s1_s2_dic
        np.save(outdir+'slope_s1_s2',slope_dic)


        pass


    def check_slope(self):

        f = self.this_class_arr+'slope\\slope_s1_s2.npy'
        dic = dict(np.load(f).item())
        for key in dic:
            one_year_dic = dic[key]
            s1_dic = {}
            s2_dic = {}
            for pix in one_year_dic:
                s1_s2 = one_year_dic[pix]
                s1_dic[pix] = s1_s2[0]
                s2_dic[pix] = s1_s2[1]
            s1_arr = DIC_and_TIF().pix_dic_to_spatial_arr(s1_dic)
            s2_arr = DIC_and_TIF().pix_dic_to_spatial_arr(s2_dic)
            plt.imshow(s1_arr)
            plt.colorbar()
            plt.figure()
            plt.imshow(s2_arr)
            plt.colorbar()
            plt.show()


    def SOS_EOS_S1_S2_S3(self):
        out_dir = self.this_class_arr+'SOS_EOS_S1_S2_S3\\'
        Tools().mk_dir(out_dir)
        fdir = this_root + r'outdir_2020\arr\Phenology\hants_smooth\\'
        vals_dic = {}
        for y in os.listdir(fdir):
            one_year_dic = {}
            for f in os.listdir(fdir + y):
                dic = dict(np.load(fdir + y + '\\' + f).item())
                one_year_dic.update(dic)
            vals_dic[y] = one_year_dic

        slope_dir = self.this_class_arr+'slope\\slope\\'

        SOS_dir = this_root+r'outdir_2020\arr\Phenology\SOS_EOS\threshold_0.5\\'
        sos_dic = {}
        for f in os.listdir(SOS_dir):
            year = f.split('.')[0]
            dic = dict(np.load(SOS_dir+f).item())
            sos_dic[year] = dic
        s1_s2_point_f = this_root+r'outdir_2020\arr\Phenology\slope\slope_s1_s2.npy'
        s1_s2_point_dic = dict(np.load(s1_s2_point_f).item())

        period_dic = {}
        for y in tqdm(sos_dic):
            sos_dic_year_dic = sos_dic[y]
            s1_s2_point_year_dic = s1_s2_point_dic[y]
            vals_year_dic = vals_dic[y]
            # slope_year_dic = dict(np.load(slope_dir+y+'.npy').item())
            period_year_dic = {}
            for pix in sos_dic_year_dic:

                # vals = vals_year_dic[pix]
                # if max(vals) < 0300:
                #     continue
                # slope = slope_year_dic[pix]
                if not pix in s1_s2_point_year_dic:
                    continue
                s1_s2_point = s1_s2_point_year_dic[pix]
                sos = sos_dic_year_dic[pix]

                p1 = sos[0]
                p2 = s1_s2_point[0]
                p3 = s1_s2_point[1]
                p4 = sos[1]

                # if
                # if p2 > p1 and p3 > p2 and p4 > p3:
                #     continue
                # if not pix == '064.038':
                #     continue
                # print p1,p2,p3,p4,pix # 064.038

                indx_list = [p1,p2,p3,p4]
                indx_list.sort()
                p1_start,p1_end = indx_list[0],indx_list[1]
                p2_start,p2_end = indx_list[1],indx_list[2]
                p3_start,p3_end = indx_list[2],indx_list[3]

                period1 = p1_start,p1_end
                period2 = p2_start,p2_end
                period3 = p3_start,p3_end
                period_year_dic[pix] = (period1,period2,period3)

                ###########        plot       ################
                # plt.plot(vals)
                # plt.plot(period1,[vals[period1[0]],vals[period1[1]]],linewidth=4,c='r')
                # plt.plot(period2,[vals[period2[0]],vals[period2[1]]],linewidth=4,c='g')
                # plt.plot(period3,[vals[period3[0]],vals[period3[1]]],linewidth=4,c='b')
                #
                # plt.twinx()
                # plt.plot(slope,'--',c='black')
                # plt.show()
                # print

            period_dic[y] = period_year_dic
        np.save(out_dir+'period_dic',period_dic)
        pass

    def s1_s2_s3_to_tif(self):
        out_tif_dir = self.this_class_tif+'s1_s2_s3_to_tif\\'
        Tools().mk_dir(out_tif_dir)
        f = self.this_class_arr + 'SOS_EOS_S1_S2_S3\\period_dic.npy'
        dic = Tools().load_npy(f)
        # year_range = range(1992,2004)
        for year in tqdm(dic):
            one_year_dic = dic[year]
            s1_dic = {}
            s2_dic = {}
            s3_dic = {}
            for pix in one_year_dic:
                s1,s2,s3 = one_year_dic[pix]
                s1_l = s1[1]-s1[0]
                s2_l = s2[1]-s2[0]
                s3_l = s3[1]-s3[0]

                s1_dic[pix] = s1_l
                s2_dic[pix] = s2_l
                s3_dic[pix] = s3_l
            DIC_and_TIF().pix_dic_to_tif(s1_dic,out_tif_dir+'{}_s1.tif'.format(year))
            DIC_and_TIF().pix_dic_to_tif(s2_dic,out_tif_dir+'{}_s2.tif'.format(year))
            DIC_and_TIF().pix_dic_to_tif(s3_dic,out_tif_dir+'{}_s3.tif'.format(year))
        pass


    def s1_s2_s3_to_tif_part12(self):

        fdir = self.this_class_tif+'s1_s2_s3_to_tif\\'
        outdir = self.this_class_tif+'s1_s2_s3_to_tif_part12\\'
        Tools().mk_dir(outdir)
        part = 2
        if part == 1:
            year_range = range(1992,2004)
        elif part == 2:
            year_range = range(2004,2016)
        else:
            raise IOError

        for s in ['s1', 's2', 's3']:
            arr_sum = 0.
            for y in year_range:
                fname = fdir + '{}_{}.tif'.format(y,s)
                arr = to_raster.raster2array(fname)[0]
                Tools().mask_999999_arr(arr)
                arr_sum += arr
            arr_mean = arr_sum/len(year_range)
            DIC_and_TIF().arr_to_tif(arr_mean,outdir+'{}_part{}.tif'.format(s,part))


    def s1_s2_s3_to_mon(self):
        outdir = self.this_class_arr+'s1_s2_s3_to_mon\\'
        Tools().mk_dir(outdir)
        f = self.this_class_arr+'SOS_EOS_S1_S2_S3\\period_dic.npy'
        period_dic = dict(np.load(f).item())
        results = {}
        for y in tqdm(period_dic):
            dic_y = period_dic[y]
            one_year_result = {}
            for pix in dic_y:
                periods = dic_y[pix]
                p1,p2,p3 = periods
                p1_start,p1_end = p1
                p2_start,p2_end = p2
                p3_start,p3_end = p3
                # print p1_start,p1_end
                init_date = datetime.datetime(int(y),1,1)
                p1_start_date = init_date+datetime.timedelta(int(p1_start))
                p1_end_date = init_date+datetime.timedelta(int(p1_end))
                # p2_start_date = init_date+datetime.timedelta(int(p2_start))
                # p2_end_date = init_date+datetime.timedelta(int(p2_end))
                # print p2_start_date.month, p2_end_date.month
                p3_start_date = init_date+datetime.timedelta(int(p3_start))
                p3_end_date = init_date+datetime.timedelta(int(p3_end))
                # print p3_start_date.month, p3_end_date.month
                # print

                p1_mon = [p1_start_date.month]
                p3_mon = range(p3_start_date.month,p3_end_date.month+1)
                p2_mon = range(p1_start_date.month,p3_end_date.month+1)
                p2_mon.remove(p1_mon[0])
                for i in p3_mon:
                    p2_mon.remove(i)
                one_year_result[pix] = [p1_mon,p2_mon,p3_mon]
            results[y]=one_year_result

        np.save(outdir+'s1_s2_s3_to_mon',results)
        pass

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
            for y in range(1992,2004):
                year_range.append(str(y))
        elif part == 2:
            for y in range(2004,2016):
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
        threshold = 0.5
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

    def start_end_part1_part2_change(self):
        fdir = self.this_class_tif+'SOS_EOS\\'
        start_part1_f = fdir+'start_part1_0.5.tif'
        start_part2_f = fdir+'start_part2_0.5.tif'
        end_part1_f = fdir+'end_part1_0.5.tif'
        end_part2_f = fdir+'end_part2_0.5.tif'
        duration_part1_f = fdir+'duration_part1_0.5.tif'
        duration_part2_f = fdir+'duration_part2_0.5.tif'

        start1_arr = to_raster.raster2array(start_part1_f)[0]
        start2_arr = to_raster.raster2array(start_part2_f)[0]
        end1_arr = to_raster.raster2array(end_part1_f)[0]
        end2_arr = to_raster.raster2array(end_part2_f)[0]
        duration1_arr = to_raster.raster2array(duration_part1_f)[0]
        duration2_arr = to_raster.raster2array(duration_part2_f)[0]

        start1_arr = Tools().mask_999999_arr(start1_arr)
        start2_arr = Tools().mask_999999_arr(start2_arr)
        end1_arr = Tools().mask_999999_arr(end1_arr)
        end2_arr = Tools().mask_999999_arr(end2_arr)
        duration1_arr = Tools().mask_999999_arr(duration1_arr)
        duration2_arr = Tools().mask_999999_arr(duration2_arr)

        start_diff = start2_arr-start1_arr
        end_diff = end2_arr-end1_arr
        duration_diff = duration2_arr-duration1_arr


        start_total_pix_num = len(start_diff[start_diff>-9999])
        end_total_pix_num = len(end_diff[end_diff>-9999])
        duration_total_pix_num = len(duration_diff[duration_diff>-9999])
        print len(start_diff[start_diff>0])/float(start_total_pix_num)
        print len(end_diff[end_diff>0])/float(end_total_pix_num)
        print len(duration_diff[duration_diff>0])/float(duration_total_pix_num)

        # plt.imshow(start_diff)
        # plt.title('start')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(end_diff)
        # plt.title('end')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(duration_diff)
        # plt.title('duraton')
        # plt.colorbar()
        #
        # plt.show()



        pass

    def s1_s2_s3_anomaly(self):
        fdir = self.this_class_tif+'s1_s2_s3_to_tif\\'
        outdir = self.this_class_tif+'s1_s2_s3_anomaly\\'
        Tools().mk_dir(outdir)
        for s in ['s1','s2','s3']:
            outdir_i = outdir+'{}\\'.format(s)
            Tools().mk_dir(outdir_i)
            s_arr = []
            for y in range(1992, 2016):
                fname = '{}_{}.tif'.format(y,s)
                arr = to_raster.raster2array(fdir+fname)[0]
                arr = np.array(arr,dtype=np.float)
                Tools().mask_999999_arr(arr)
                s_arr.append(arr)
            s_arr = np.array(s_arr)
            spatial_mean,spatial_std = Tools().arrs_mean_std(s_arr)
            for i in range(len(s_arr)):
                arr_i = s_arr[i]
                arr_anomaly = (arr_i-spatial_mean)/spatial_std
                fname = '{}.tif'.format(i+1992)
                DIC_and_TIF().arr_to_tif(arr_anomaly,outdir_i+fname)

    def SOS_EOS_Duration_to_tif(self):
        fdir = self.this_class_arr+r'SOS_EOS\threshold_0.5\\'
        outdir = self.this_class_tif+'SOS_EOS_Duration_to_tif\\'
        Tools().mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            year = f.split('.')[0]
            dic = Tools().load_npy(fdir+f)
            start_dic = {}
            end_dic = {}
            duration_dic = {}
            for pix in dic:
                start,end,duration = dic[pix]
                start_dic[pix] = start
                end_dic[pix] = end
                duration_dic[pix] = duration
            start_arr = DIC_and_TIF().pix_dic_to_spatial_arr(start_dic)
            end_arr = DIC_and_TIF().pix_dic_to_spatial_arr(end_dic)
            duration_arr = DIC_and_TIF().pix_dic_to_spatial_arr(duration_dic)
            DIC_and_TIF().arr_to_tif(start_arr,outdir+year+'_SOS.tif')
            DIC_and_TIF().arr_to_tif(end_arr,outdir+year+'_EOS.tif')
            DIC_and_TIF().arr_to_tif(duration_arr,outdir+year+'_duration.tif')
        pass

    def SOS_EOS_Duration_anomaly(self):
        fdir = self.this_class_tif + 'SOS_EOS_Duration_to_tif\\'
        outdir = self.this_class_tif + 'SOS_EOS_Duration_anomaly\\'
        Tools().mk_dir(outdir)
        for s in ['SOS', 'EOS', 'duration']:
            outdir_i = outdir + '{}\\'.format(s)
            Tools().mk_dir(outdir_i)
            s_arr = []
            for y in range(1992, 2016):
                fname = '{}_{}.tif'.format(y, s)
                arr = to_raster.raster2array(fdir + fname)[0]
                arr = np.array(arr, dtype=np.float)
                Tools().mask_999999_arr(arr)
                s_arr.append(arr)
            s_arr = np.array(s_arr)
            spatial_mean, spatial_std = Tools().arrs_mean_std(s_arr)
            for i in range(len(s_arr)):
                arr_i = s_arr[i]
                arr_anomaly = (arr_i - spatial_mean) / spatial_std
                fname = '{}.tif'.format(i + 1992)
                DIC_and_TIF().arr_to_tif(arr_anomaly, outdir_i + fname)

        pass

class Phenology:

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Phenology\\'
        self.this_class_tif = results_root + 'tif\\Phenology\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        # 1 把多年的NDVI分成单年，分文件夹存储
        # 2 把单年的NDVI tif 转换成 perpix
        # self.data_transform_split_files()
        # 3 hants smooth
        # self.hants()
        self.check_hants()
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


    def data_transform_split_files(self):
        fdir = data_root + 'NDVI\\tif_0.5_bi_weekly_yearly\\'
        outdir = data_root + 'NDVI\\per_pix_yearly\\'
        for year in os.listdir(fdir):
            print year
            fdir_i = fdir + year + '\\'
            outdir_i = outdir + year + '\\'
            Tools().mk_dir(outdir_i, force=1)
            self.data_transform(fdir_i, outdir_i)

        pass


    def kernel_hants(self,params):
        outdir,y,fdir = params
        outdir_y = outdir + y + '\\'
        Tools().mk_dir(outdir_y, force=1)
        for f in os.listdir(fdir + y):
            dic = dict(np.load(fdir + y + '\\' + f).item())
            hants_dic = {}
            for pix in dic:
                vals = dic[pix]
                vals = np.array(vals)
                std = np.std(vals)
                if std == 0:
                    continue
                xnew, ynew = self.__interp__(vals)
                ynew = np.array([ynew])
                # print np.std(ynew)
                results = HANTS(sample_count=365, inputs=ynew, low=-10000, high=10000,
                                fit_error_tolerance=std)
                result = results[0]

                # plt.plot(result)
                # plt.plot(range(len(ynew[0])),ynew[0])
                # plt.show()
                hants_dic[pix] = result
            np.save(outdir_y + f, hants_dic)



    def hants(self):
        outdir = self.this_class_arr+'hants_smooth\\'
        fdir = data_root+'NDVI\\per_pix_yearly\\'
        params = []
        for y in os.listdir(fdir):
            params.append([outdir,y,fdir])
        MULTIPROCESS(self.kernel_hants,params).run(process=1)


    def check_hants(self):

        fdir = self.this_class_arr+'hants_smooth\\'
        for year in os.listdir(fdir):
            perpix_dir = fdir+'{}\\'.format(year)
            for f in os.listdir(perpix_dir):
                if not '015' in f:
                    continue
                dic = T.load_npy(perpix_dir+f)
                for pix in dic:
                    vals = dic[pix]
                    if len(vals) > 0:
                        # print pix,vals
                        plt.plot(vals)
                        plt.show()
                        sleep()
            exit()



    def __interp__(self,vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        x_new = np.linspace(min(inx),max(inx),365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new,y_new


def main():
    Phenology().run()
    pass


if __name__ == '__main__':
    main()