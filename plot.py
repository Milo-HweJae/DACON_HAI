def make_plotly(y_pred_disc, y_pred_ae, y_true,th_ae,th_disc):
    import pickle
    import plotly.offline as py
    import plotly.graph_objs as go
    import numpy as np
    
    # 경로에 각자 y_store.pkl 파일 경로 입력 & atk_range2.pkl 파일 ./에 위치 시키기
    # f1 = open('D:\\Research Project\\2020_nsr_ids\\main\\y_store.pkl', 'rb')
    # f2 = open('D:\\Research Project\\2020_nsr_ids\\main\\normal_.pkl', 'rb')
    atk_range = pickle.load(open('atk_range.pkl', 'rb'))
    
    # d : 0초부터 시작해서 index
    d=[]
    # c, d = pickle.load(f2)
    for i in range(atk_range[-1][2]):
        d.append(i)
    data1 = go.Scatter(x=d, y=y_pred_disc[:len(d)], line=dict(color='black', width=1))
    data2 = go.Scatter(x=d, y=y_pred_ae[:len(d)], line=dict(color='black', width=1))
    # data2 = go.Scatter(x=c, y=d)
    
    
    th = th_disc
    th1 = []
    th2 = []
    for i in range(len(d)):
       th1.append(th)
    
    th = th_ae
    for i in range(len(d)):
       th2.append(th)
    
    th1 = go.Scatter(x=d, y=th1, line=dict(color='royalblue', width=1, dash='dash'))
    th2 = go.Scatter(x=d, y=th2, line=dict(color='royalblue', width=1, dash='dash'))
    
    
    
    atk_range2 = pickle.load(open('atk_range.pkl', 'rb'))
    
    data=[]
    data.append(data1)
    data.append(th1)
    # data = [data1, th1]
    for i in range(76):
        y1 = []
        if i % 2 ==0:        
            for j in range(1000): 
                y1.append(atk_range2[int(i/2)][0])
                if j == 999:
                    data.append(go.Scatter(y = np.arange(0.0,1.0,0.001), x = y1, line=dict(color='firebrick', width=1, dash='dash')))
        else:
            for j in range(1000):
                y1.append(atk_range2[int((i-1)/2)][1])
                if j == 999: 
                    data.append(go.Scatter(y = np.arange(0.0,1.0,0.001), x = y1, line=dict(color='firebrick', width=1, dash='dash')))
    py.plot(data, filename='./imgs/disc.html')
    data=[]
    data.append(data2)
    data.append(th2)
    for i in range(76): 
        y1 = []
        if i % 2 ==0:        
            for j in range(1000): 
                y1.append(atk_range2[int(i/2)][0])
                if j == 999:
                    data.append(go.Scatter(y = np.arange(0.0,1.0,0.001), x = y1, line=dict(color='firebrick', width=1, dash='dash')))
        else:
            for j in range(1000):
                y1.append(atk_range2[int((i-1)/2)][1])
                if j == 999: 
                    data.append(go.Scatter(y = np.arange(0.0,1.0,0.001), x = y1, line=dict(color='firebrick', width=1, dash='dash')))
       
    py.plot(data, filename='./imgs/ae.html')
    # pyo.iplot(data2)
    
    
    # f1.close()
    # f2.close()
    
def plot_save(timestamp, y_pred_ae,y_pred_disc, y_true, th_ae, th_disc, ylim_ae, ylim_disc):
    import pickle
    import matplotlib.pyplot as plt
    # 공격 구간 별 탐지 결과 이미지 저장
    IMG_DIR = r'./imgs/'
    atk_range = pickle.load(open('atk_range.pkl', 'rb'))
    
    timestamp_ = []
    for item in timestamp:
        timestamp_.extend(item)
    # discriminator 결과 저장
    i = 1
    for start, end, next_start in atk_range:
        if next_start - end >= 3000:
            next_start = end + 3000
        
        plt.figure()
        plt.plot(range(start-1500, next_start), y_pred_disc[start-1500:next_start], linewidth=0.5)
        plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
        plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
        plt.axhline(y=th_disc, color='k', linestyle='--', linewidth=0.5)
        plt.ylim(ylim_disc)
        plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
        plt.title('Discriminator, Attack{}'.format(i))
        plt.savefig('./imgs/attack/disc/attack{}.png'.format(i), dpi=300)
        plt.close()
        i += 1
    # autoencoder 결과 저장
    i = 1
    for start, end, next_start in atk_range:
        if next_start - end >= 3000:
            next_start = end + 3000
        plt.figure()
        plt.plot(range(start-1500, next_start), y_pred_ae[start-1500:next_start], linewidth=0.5)
        plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
        plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
        plt.yscale('log')
        plt.ylim(ylim_ae)
        plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
        plt.axhline(y=th_ae, color='k', linestyle='--', linewidth=0.5)
        plt.title('AE, Attack{}'.format(i))
        # plt.savefig('./imgs/attack/ae/attack{}.png'.format(i), dpi=300)
        plt.savefig('./imgs/attack/ae/attack{}.png'.format(i), dpi=300)
        plt.close()
        i += 1
        
    # disc + ae 결과 저장
    # i = 1
    # for start, end, next_start in atk_range:
    #     if next_start - end >= 3000:
    #         next_start = end + 3000
    #     interval = int((next_start - start + 1500)/5)
    #     plt.figure()
    #     plt.subplot(122)
    #     plt.plot(range(start-1500, next_start), y_pred_disc[start-1500:next_start], linewidth=0.5)
    #     plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
    #     plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
    #     plt.axhline(y=th_disc, color='r', linestyle='--', linewidth=0.5)
    #     plt.ylim(ylim_disc)
    #     plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
    #     plt.title('Discriminator')
    #     plt.subplot(121)
    #     plt.plot(range(start-1500, next_start), y_pred_ae[start-1500:next_start], linewidth=0.5)
    #     plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
    #     plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
    #     plt.yscale('log')
    #     plt.ylim(ylim_ae)
    #     plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_ae, color='k', linestyle='--',linewidth=0.5)
    #     plt.title('AE')
    #     plt.suptitle('Attack{}'.format(i))
    #     plt.savefig('./imgs/plus/attack/disc+ae/disc+ae_{}.png'.format(i), dpi=300)
    #     plt.close()
    #     i += 1
    
    # sub_ae 결과 저장
    # i = 1
    # for start, end, next_start in atk_range:
    #     if next_start - end >= 3000:
    #         next_start = end + 3000
    #     interval = int((next_start - start + 1500)/5)
    #     plt.figure()
    #     plt.subplot(221)
    #     plt.plot(range(start-1500, next_start), y_pred_sub[0][start-1500:next_start], linewidth=0.5)
    #     plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
    #     plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
    #     plt.axhline(y=th_sub[0], color='r', linestyle='--', linewidth=0.5)
    #     plt.ylim([ylim_ae[0], ylim_ae[1]])
    #     plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
    #     plt.title('Process 1')
        
    #     plt.subplot(222)
    #     plt.plot(range(start-1500, next_start), y_pred_sub[1][start-1500:next_start], linewidth=0.5)
    #     plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
    #     plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
    #     plt.axhline(y=th_sub[1], color='r', linestyle='--', linewidth=0.5)
    #     plt.ylim([1e-7, 1e-5])
    #     plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
    #     plt.title('Process 2')
        
        
    #     plt.subplot(223)
    #     plt.plot(range(start-1500, next_start), y_pred_sub[2][start-1500:next_start], linewidth=0.5)
    #     plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
    #     plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
    #     plt.axhline(y=th_sub[2], color='r', linestyle='--', linewidth=0.5)
    #     plt.ylim([1e-7, 1e-5])
    #     plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
    #     plt.title('Process 3')
        
        
    #     plt.subplot(224)
    #     plt.plot(range(start-1500, next_start), y_pred_sub[3][start-1500:next_start], linewidth=0.5)
    #     plt.axvline(x=start, color='r', linestyle='--', linewidth=0.5)
    #     plt.axvline(x=end, color='r', linestyle='--', linewidth=0.5)
    #     plt.axhline(y=th_sub[3], color='r', linestyle='--', linewidth=0.5)
    #     plt.ylim([1e-6, 6e-5])
    #     plt.xticks([start-1500,start,end, next_start],[timestamp_[start-1500],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
    #     plt.title('Process 4')
        
        
    #     plt.suptitle('Attack{}'.format(i))
    #     plt.savefig('./imgs/plus/attack/sub_ae/sub_ae_{}.png'.format(i), dpi=300)
    #     plt.close()
    #     i += 1
        
        
    # normal 구간 결과 저장
    i = 0
    for start, end, next_start in atk_range:
        plt_start = i
        plt_end = start
        
        # Disc result
        plt.figure()
        plt.plot(range(plt_start, plt_end), y_pred_disc[plt_start:plt_end], linewidth=0.5)
        plt.ylim(ylim_disc)
        plt.xticks([plt_start,start,end,next_start],[timestamp_[plt_start],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
        plt.axhline(y=th_disc, color='k', linewidth=1)
        plt.title('Discriminator, Normal{}-{}'.format(plt_start, plt_end))
        plt.savefig('./imgs/normal/disc/normal{}-{}.png'.format(plt_start, plt_end), dpi=300)
        plt.close()
        
        # TR result
        plt.figure()
        plt.plot(range(plt_start, plt_end), y_pred_ae[plt_start:plt_end], linewidth=0.5)
        plt.ylim(ylim_ae)
        plt.xticks([plt_start,start,end,next_start],[timestamp_[plt_start],timestamp_[start],timestamp_[end],timestamp_[next_start]],rotation=30,fontsize=4)
        plt.axhline(y=th_ae, color='r', linewidth=1)
        plt.title('AE, Normal{}-{}'.format(plt_start, plt_end))
        plt.savefig('./imgs/normal/ae/normal{}-{}.png'.format(plt_start, plt_end), dpi=300)
        plt.close()
        
        i = end

    # normal 구간 결과 저장
    # i = 0
    # for start, end, next_start in atk_range:
    #     plt_start = i
    #     plt_end = start
        
    #     # Disc + ae result
    #     plt.figure()
    #     plt.subplot(122)
    #     plt.plot(range(plt_start, plt_end), y_pred_disc[plt_start:plt_end], linewidth=0.5)
    #     plt.ylim(ylim_disc)
    #     plt.xticks([plt_start,plt_end],[timestamp_[plt_start],timestamp_[plt_end]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_disc, color='r', linewidth=1)
    #     plt.title('Discriminator')
        
    #     plt.subplot(121)
    #     plt.plot(range(plt_start, plt_end), y_pred_ae[plt_start:plt_end], linewidth=0.5)
    #     plt.ylim(ylim_ae)
    #     plt.xticks([plt_start,plt_end],[timestamp_[plt_start],timestamp_[plt_end]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_ae, color='r', linewidth=1)
    #     plt.title('AE')
    #     plt.suptitle('Normal{}-{}'.format(plt_start, plt_end))
    #     plt.savefig('./imgs/plus/normal/disc+ae/disc+ae{}-{}.png'.format(plt_start, plt_end), dpi=300)
    #     plt.close()
    
    #     i = end

    # normal 구간 sub_ae 결과 저장
    # i = 0
    # for start, end, next_start in atk_range:
    #     plt_start = i
    #     plt_end = start
        
        
    #     plt.figure()
    #     plt.subplot(221)
    #     plt.plot(range(plt_start, plt_end), y_pred_sub[0][plt_start:plt_end], linewidth=0.5)
    #     plt.ylim([ylim_ae[0], ylim_ae[1]])
    #     plt.xticks([plt_start,plt_end],[timestamp_[plt_start],timestamp_[plt_end]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_sub[0], color='r', linewidth=1)
    #     plt.title('Process 1')
        
    #     plt.subplot(222)
    #     plt.plot(range(plt_start, plt_end), y_pred_sub[1][plt_start:plt_end], linewidth=0.5)
    #     plt.ylim([1e-7, 1e-5])
    #     plt.xticks([plt_start,plt_end],[timestamp_[plt_start],timestamp_[plt_end]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_sub[1], color='r', linewidth=1)
    #     plt.title('Process 2')
        
        
    #     plt.subplot(223)
    #     plt.plot(range(plt_start, plt_end), y_pred_sub[2][plt_start:plt_end], linewidth=0.5)
    #     plt.ylim([1e-7, 1e-5])
    #     plt.xticks([plt_start,plt_end],[timestamp_[plt_start],timestamp_[plt_end]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_sub[2], color='r', linewidth=1)
    #     plt.title('Process 3')
        
        
    #     plt.subplot(224)
    #     plt.plot(range(plt_start, plt_end), y_pred_sub[3][plt_start:plt_end], linewidth=0.5)
    #     plt.ylim([1e-6, 6e-5])
    #     plt.xticks([plt_start,plt_end],[timestamp_[plt_start],timestamp_[plt_end]],rotation=30,fontsize=4)
    #     plt.axhline(y=th_sub[3], color='r', linewidth=1)
    #     plt.title('Process 4')
        
        
    #     plt.suptitle('Normal{}-{}'.format(plt_start, plt_end))
    #     plt.savefig('./imgs/plus/normal/sub_ae/sub_ae_{}.png'.format(i), dpi=300)
    #     plt.close()
        
    #     i = end
        

def eval_disc(prediction, test_label, threshold):
    return evaluate(1, prediction, test_label, threshold)

def eval_ae(prediction, test_label, threshold):
    return evaluate(9999, prediction, test_label, threshold)

def evaluate(ylim, prediction, test_label, threshold):
    import matplotlib.pyplot as plt
    import numpy as np
    # import plotly.offline as pyo
    import pickle
    ## prediction : numpy.array형, 0-1 사이값 N개 [N,]
    ## test_label : 비교할 label
    ## threshold : 임계값, prediction이 임계값 이상이면 공격, 아니면 정상상태
    
    flag=1
    i=0
    j=0
    m=0
    t1=0
    t2=0
    
    ## prediction_sec는 prediction 길이(초)
    prediction_sec = np.zeros(len(prediction))
    prediction_plot = prediction.astype(str) 
    
    ## y는 test_label 길이(초)
    test_label_sec = np.zeros(len(test_label))
    
    ## test_label 공격있었던거 label 달기 위한 str배열
    # P1
    test_label_p1=np.zeros(len(test_label))
    test_label_p1=test_label_p1.astype(str)
    for i in range(0,len(test_label),1):
        if test_label[i][1]==1:
            test_label_p1[i]='P1'
        else:
            test_label_p1[i]='Attack'         
    #P2
    i=0
    test_label_p2=np.zeros(len(test_label))
    test_label_p2=test_label_p2.astype(str)
    for i in range(0,len(test_label),1):
        if test_label[i][2]==1:
            test_label_p2[i]='P2'
        else:
            test_label_p2[i]='Attack'         
    #P3
    i=0        
    test_label_p3=np.zeros(len(test_label))
    test_label_p3=test_label_p3.astype(str)
    for i in range(0,len(test_label),1):
        if test_label[i][3]==1:
            test_label_p3[i]='P3'
        else:
            test_label_p3[i]='Attack'         
    #Total
    i=0        
    test_label_total=np.zeros(len(test_label))
    test_label_total=test_label_total.astype(str)
    for i in range(0,len(test_label),1):
        if test_label[i][0]==1:
            test_label_total[i]='Attack'
        else:
            test_label_total[i]='Normal' 
            
    while True:
        test_label_sec[m]=m
        if m >= len(test_label_sec)-1:
            break
        m=m+1
    
    i=0
    while True:
        
        if i>=len(prediction):
            break
        ## prediction_sec = 1초 간격(0초부터 시작), prediction[i]는 공격이 탐지되면 1 아니면 0
        if prediction[i] >= threshold:
            prediction_sec[i] = i
        else:
            prediction_sec[i] = i
        
        ## i는 1(초)씩 증가
        i=i+1
        j=j+1
    
    i=0
    temp_p1_sec=[]
    temp_p1_label=[]
    temp_p2_sec=[]
    temp_p2_label=[]
    temp_p3_sec=[]
    temp_p3_label=[]
    
    
    for i in range(len(test_label)):
        if test_label[i][1] == 1:
            temp_p1_sec.append(i)
            temp_p1_label.append('P1')
        if test_label[i][2] == 1:
            temp_p2_sec.append(i)
            temp_p2_label.append('P2')
        if test_label[i][3] == 1:
            temp_p3_sec.append(i)
            temp_p3_label.append('P3')  
        
        
    
    ## plot 그리기(실제 공격 주입 그래프)
    plt.figure()
    plt.subplot(211)
    plt.xlim(0,len(test_label)+4)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,3),useMathText=True)
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(np.arange(5),("Normal","Attack","P1","P2","P3"))
    plt.plot(test_label_sec,test_label_total,'y')
    plt.plot(temp_p3_sec,temp_p3_label,'c_')
    plt.plot(temp_p2_sec,temp_p2_label,'g_')
    plt.plot(temp_p1_sec,temp_p1_label,'b_')
    plt.xlabel('Time(sec)')
    plt.ylabel('Attack')
    plt.subplots_adjust(hspace=0.3)
    plt.show
    
    data = (test_label_sec, test_label_total)
    # data = [test_label_sec, test_label_total]
    # pyo.iplot(data) 
    # with open('D:\\Research Project\\2020_nsr_ids\\main\\attack_.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    
    ## plot 그리기(모델이 예측한 공격 탐지 그래프)
    plt.subplot(212)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,3),useMathText=True)
    plt.xticks(rotation=45, fontsize=7)
    if ylim != 9999:
        plt.ylim(0,ylim)
    else:
        plt.yscale("log")
    plt.xlim(0,len(test_label)+4)
    plt.plot(prediction_sec,prediction,'y')
    temp_threshold=np.zeros(len(prediction))
    for xx in range(len(temp_threshold)):
        temp_threshold[xx]=threshold
    plt.plot(prediction_sec,temp_threshold,'m:')
    plt.xlabel('Time(sec)')
    plt.ylabel('Anomaly\ndetection\n')
    plt.show
    
    # data = (prediction_sec, prediction)
    # with open('D:\\Research Project\\2020_nsr_ids\\main\\normal_.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # data = [prediction_sec, prediction]
    # pyo.iplot(data)
    
    ## i는 time
    i=0
    
    ## j는 실제 공격 개수
    j=0
    
    ## k는 실제 탐지 개수
    k=0
    
    t1_array=[]
    t2_array=[]

    
    
    
    ## 실제 공격 기록
    while flag:
    
        ## 1(공격)이 주입됨
        if test_label[i][0]==1:
            t1_array.append(i)
            ## 공격이 끝날때까지 시간 계속 증가
            while test_label[i][0]!=0:
                if i>=(len(test_label)-1):
                    
                    break
                
                i=i+1
            if t1_array[-1]==len(test_label)-1:
                t2_array.append(len(test_label)-1)
            else:
                t2_array.append(i-1)
            ## 공격 개수 count
            j=j+1
            
        
        i=i+1
        
        ## 끝까지 보면 while 탈출
        if i>=len(test_label):
            flag=0
    
    ## 실제 전체 공격의 개수 출력
    print("전체 공격의 개수 : " + str(j) + "개\n")
    
    ## 실제 공격 개수만큼 배열 생성
    delta_t=np.zeros(j)
    
    
    ## 탐지 공격 기록       
    
    ## while 탈출용 변수
    flag=1
    ## 시간용 변수 1,2,3, ...
    i=0
    ## 몇번째 공격을 탐지했는지를 위한 변수, ex)m=1 1번째 공격
    m=0
    
    temp=0
    
    while flag:    
    
   
    ## 공격이 발생한 시간동안 감지됐는지
        ## 공격 시작 시간
        t=t1_array[m]
        
        
        ## t1에서 t2사이 동안
        while t>=t1_array[m] and t<=t2_array[m]:
            
            ## 공격이 탐지됐으면 탐지된거 count, 그때까지 걸린 시간 저장
            if prediction[t] >= threshold:
                k=k+1
                delta_t[m]=t-t1_array[m]
                ## 공격 시작되자마자 탐지(간격 0초)일때를 위함
                if t-t1_array[m]==0:
                    delta_t[m]=-5
                break
            
            ## 탐지 안됐으면 해당 공격 번호에 -1 저장
            else:
                if t==t2_array[m]:
                    delta_t[m]=-1
            t=t+1
        m=m+1
                    
                
        if m>=len(t1_array):
            flag=0
    
    ## 탐지한 공격 개수 출력        
    print("탐지한 공격의 개수 : " + str(k) + "개\n")    
    
    
    ## 임시 변수, l은 탐지 못한 공격 개수 count, i는 각 공격 탐지했는지 못했는지를 체크하기 위한 순서 변수
    l=0
    i=0
    
    ## print(delta_t)
    
    ## 각 공격 발생 후 탐지 될 때까지 걸린 시간 출력
    for temp in delta_t:
        if temp != 0:
            if temp != -1:
                if temp == -5:
                    temp=0
                print(str(i+1)+"번째 공격 발생 후 탐지될 때까지 걸린 시간 : "+str(temp)+'초\n')
                
        if temp == -1:
            l=l+1
        i=i+1
    
    ## 탐지 못한 공격 개수 & 정탐율 출
    print("탐지 못한 공격의 개수 : "+str(l) + "개\n")
    print("정탐율 : " + str(100*k/j) + "%\n")
      
        
          
## TEST ##        

# rnd = lambda t: round(t)
# test_label = np.random.random([10, 4])
# test_label = np.vectorize(rnd)(test_label)
# prediction = np.random.random(10)

# threshold = 0.9

# evaluate(prediction, test_label, threshold)       
      
        
          
## TEST ##        

# rnd = lambda t: round(t)
# test_label = np.random.random([10, 4])
# test_label = np.vectorize(rnd)(test_label)
# prediction = np.random.random(10)

# threshold = 0.9

# evaluate(prediction, test_label, threshold)          

def tapr(y_pred_disc, y_pred_ae, y_pred_sub, y_true, th_ae, th_sub, th_disc):
    import numpy as np
    from TaPR_pkg import etapr
    from sklearn.metrics import f1_score, recall_score, precision_score

    y_true_0 = y_true[:,0]
    # y_true_p1, y_true_p2, y_true_p3 = y_true[:,1], y_true[:,2], y_true[:,3]
    
    y_pred_ae[y_pred_ae > th_ae] = 1
    y_pred_ae[y_pred_ae <= th_ae] = 0
    
    # for i in range(len(y_pred_sub)):
    #     y_pred_sub[i][y_pred_sub[i] > th_sub[i]] = 1
    #     y_pred_sub[i][y_pred_sub[i]<= th_sub[i]] = 0
        
    # y_pred_disc[y_pred_disc > th_disc] = 1
    # y_pred_disc[y_pred_disc <= th_disc] = 0
    
    # y_pred = np.vstack((y_pred_ae, y_pred_disc))
    # y_pred = y_pred.max(axis=0)
    
    # y_pred_sub_all = np.vstack((y_pred_sub[0], y_pred_sub[1], y_pred_sub[2], y_pred_sub[3]))
    # y_pred_sub_all = y_pred_sub_all.max(axis=0)
    
    # y_pred_all = np.vstack((y_pred_ae, y_pred_disc, y_pred_sub[0], y_pred_sub[1], y_pred_sub[2], y_pred_sub[3]))
    # y_pred_all = y_pred_all.max(axis=0)
    
    
    # y_pred_all_2 = np.vstack((y_pred_ae, y_pred_sub[0], y_pred_sub[1], y_pred_sub[2], y_pred_sub[3]))
    # y_pred_all_2 = y_pred_all_2.max(axis=0)
    
    
    TaPR_ae = etapr.evaluate(anomalies=y_true_0, predictions=y_pred_ae)
    print(f"TaPR_ae_F1: {TaPR_ae['f1']:.3f} (TaPR_ae_TaP: {TaPR_ae['TaP']:.3f}, TaPR_ae_TaR: {TaPR_ae['TaR']:.3f})")
    print(f"# of detected anomalies: {len(TaPR_ae['Detected_Anomalies'])}")
    print(f"Detected anomalies: {TaPR_ae['Detected_Anomalies']}")   
    
    print('precision_ae :', precision_score(y_true_0, y_pred_ae))
    print('recall_ae :', recall_score(y_true_0, y_pred_ae))
    print('f1_score_ae :', f1_score(y_true_0, y_pred_ae))
    
    print()
    
    TaPR_disc = etapr.evaluate(anomalies=y_true_0, predictions=y_pred_disc)
    print(f"TaPR_disc_F1: {TaPR_disc['f1']:.3f} (TaPR_disc_TaP: {TaPR_disc['TaP']:.3f}, TaPR_disc_TaR: {TaPR_disc['TaR']:.3f})")
    print(f"# of detected anomalies: {len(TaPR_disc['Detected_Anomalies'])}")
    print(f"Detected anomalies: {TaPR_disc['Detected_Anomalies']}")
    
    print('precision_disc :', precision_score(y_true_0, y_pred_disc))
    print('recall_disc :', recall_score(y_true_0, y_pred_disc))
    print('f1_score_disc :', f1_score(y_true_0, y_pred_disc))
    
    print()
    
    
    # TaPR = etapr.evaluate(anomalies=y_true_0, predictions=y_pred)
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")    
        
    # print('precision :', precision_score(y_true_0, y_pred))
    # print('recall :', recall_score(y_true_0, y_pred))
    # print('f1_score :', f1_score(y_true_0, y_pred))
    
    # print()
    
    
    
    
    # TaPR = etapr.evaluate(anomalies=y_true_p1, predictions=y_pred_sub[0])
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    
    # print('precision_p1 :', precision_score(y_pred_sub[0], y_true_p1))
    # print('recall_p1 :', recall_score(y_pred_sub[0], y_true_p1))
    # print('f1_score_p1 :', f1_score(y_pred_sub[0], y_true_p1))
    
    # print()
    
    # TaPR = etapr.evaluate(anomalies=y_true_p2, predictions=y_pred_sub[1])
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    
    # print('precision_p2 :', precision_score(y_pred_sub[1], y_true_p2))
    # print('recall_p2 :', recall_score(y_pred_sub[1], y_true_p2))
    # print('f1_score_p2 :', f1_score(y_pred_sub[1], y_true_p2))
    
    # print()
    
    
    # TaPR = etapr.evaluate(anomalies=y_true_p3, predictions=y_pred_sub[2])
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    
    # print('precision_p3:', precision_score(y_pred_sub[2], y_true_p3))
    # print('recall_p3 :', recall_score(y_pred_sub[2], y_true_p3))
    # print('f1_score_p3 :', f1_score(y_pred_sub[2], y_true_p3))
    
    # print()
    
       
    
    
    # TaPR = etapr.evaluate(anomalies=y_true_0, predictions=y_pred_sub_all)
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    
    # print('precision :', precision_score(y_true_0, y_pred_sub_all))
    # print('recall :', recall_score(y_true_0, y_pred_sub_all))
    # print('f1_score :', f1_score(y_true_0, y_pred_sub_all))
    
    # print()
    
    
    
    # TaPR = etapr.evaluate(anomalies=y_true_0, predictions=y_pred_all)
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    
    # print('precision :', precision_score(y_true_0, y_pred_all))
    # print('recall :', recall_score(y_true_0, y_pred_all))
    # print('f1_score :', f1_score(y_true_0, y_pred_all))
    
    # print()
    
    
    # TaPR = etapr.evaluate(anomalies=y_true_0, predictions=y_pred_all_2)
    # print(f"TaPR_F1: {TaPR['f1']:.3f} (TaPR_TaP: {TaPR['TaP']:.3f}, TaPR_TaR: {TaPR['TaR']:.3f})")
    # print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    # print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    
    # print('precision :', precision_score(y_true_0, y_pred_all_2))
    # print('recall :', recall_score(y_true_0, y_pred_all_2))
    # print('f1_score :', f1_score(y_true_0, y_pred_all_2))
    
    # print()
    
    
    # print('precision :', precision_score(y_true_0, y_pred))
    # print('precision_ae :', precision_score(y_true_0, y_pred_ae))
    # print('precision_disc :', precision_score(y_true_0, y_pred_disc))
    # print('recall :', recall_score(y_true_0, y_pred))
    # print('recall_ae :', recall_score(y_true_0, y_pred_ae))
    # print('recall_disc :', recall_score(y_true_0, y_pred_disc))
    # print('f1_score :', f1_score(y_true_0, y_pred))
    # print('f1_score_ae :', f1_score(y_true_0, y_pred_ae))
    # print('f1_score_disc :', f1_score(y_true_0, y_pred_disc))

def result(timestamp, y_pred_ae, y_pred_disc, y_true, th_ae, th_disc, ylim_ae, ylim_disc):
    # eval_disc(y_pred_disc, y_true, threshold=th_disc)
    # eval_ae(y_pred_ae, y_true, threshold=th_ae)
    # eval_ae(y_pred_sub[0], y_true, threshold=th_ae)
    # eval_ae(y_pred_sub[1], y_true, threshold=th_ae)
    # eval_ae(y_pred_sub[2], y_true, threshold=th_ae)
    # eval_ae(y_pred_sub[3], y_true, threshold=th_ae)
    
    plot_save(timestamp, y_pred_ae, y_pred_disc, y_true, th_ae, th_disc, ylim_ae, ylim_disc)
    # make_plotly(y_pred_disc, y_pred_ae, y_true,th_ae, th_disc)
    # tapr(y_pred_ae, y_true, th_ae)