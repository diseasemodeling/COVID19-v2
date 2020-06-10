
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import pathlib
from matplotlib.ticker import FuncFormatter
from ImageMerge import merge_image_colab


### Function to plot sceanrio results
def plot_results(table):
    table_path = pathlib.Path('results/table%s.xlsx'%table)
    excel = pd.ExcelFile(table_path)
    Names = excel.parse(sheet_name = 'interventions')
    N = Names.shape[0]
   
    plt.style.use('seaborn')
    color_set = sns.color_palette('Paired',n_colors = N)
    linewidth=1

    # first graph
    fig, ax = plt.subplots(2, sharex=True)
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        
        names.append( "\n".join(name.split(",")[:2]))
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'number of infected, diagnosed',legend = False, fontsize = 10, ax = ax[0], c=color_set[i],linewidth=linewidth)
        df.plot(x = 'Date', y = 'number of infected, undiagnosed',legend = False, fontsize = 10, ax = ax[1], c=color_set[i],linewidth=linewidth)
     
    
    ax[0].set_title("Number of people with diagnosed infection", fontsize = 13)
    ax[1].set_title("Number of people with undiagnosed infection", fontsize = 13)
    ax[1].set_xlabel("Date", fontsize = 10)
   								
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.flatten()[-1].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.2),fontsize = 10)
    plt.savefig("table-%s-1.png"%(table), dpi = 200)
    plt.close()

    # second graph 
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        
        names.append( "\n".join(name.split(",")[:2]))
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'Value of statistical life-year (VSL) loss',legend = False, fontsize = 10, ax = ax, c=color_set[i],linewidth=linewidth)
       
    ax.set_title("Value of statistical life-year (VSL) loss", fontsize = 13)
    ax.set_ylabel("US dollars (in millions)",fontsize = 10)
    ax.set_xlabel("Date", fontsize = 10)
   								
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.1),fontsize = 10)
    plt.savefig("table-%s-2.png"%(table), dpi = 200)
    plt.close()
    
    # third graph 
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        
        names.append( "\n".join(name.split(",")[:2]))
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'Wage loss',legend = False, fontsize = 10, ax = ax, c=color_set[i],linewidth=linewidth)
       
    ax.set_title("Wage loss", fontsize = 13)
    ax.set_ylabel("US dollars (in millions)",fontsize = 10)
    ax.set_xlabel("Date", fontsize = 10)
   								
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.1),fontsize = 10)
    plt.savefig("table-%s-3.png"%(table), dpi = 200)
    plt.close()

    # forth graph 
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        
        names.append( "\n".join(name.split(",")[:2]))
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'total cost of testing',legend = False, fontsize = 10, ax = ax, c=color_set[i],linewidth=linewidth)
       
    ax.set_title("Total testing cost", fontsize = 13)
    ax.set_ylabel("US dollars (in millions)",fontsize = 10)
    ax.set_xlabel("Date", fontsize = 10)
   								
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.1),fontsize = 10)
    plt.savefig("table-%s-4.png"%(table), dpi = 200)
    plt.close()

    # fifth graph 
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        
        names.append( "\n".join(name.split(",")[:2]))
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'number of contact tracing test needed',legend = False, fontsize = 10, ax = ax, c=color_set[i],linewidth=linewidth)
       
    ax.set_title("Number of contact tracing needed", fontsize = 13)
    ax.set_xlabel("Date", fontsize = 10)
   								
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.1),fontsize = 10)
    plt.savefig("table-%s-5.png"%(table), dpi = 200)
    plt.close()


def plot_results_colab(table, start_d, decision_d, data):
    # actual data
    actual_data = data.parse(sheet_name ='Actual epidemic data', index_col = 0)
    actual_unemp = data.parse(sheet_name ='Actual unemployment rate', index_col = 0)
    day = pd.Timestamp(decision_d)

    table_path = pathlib.Path('results/table%s.xlsx'%table)
    excel = pd.ExcelFile(table_path)
    Names = excel.parse(sheet_name = 'interventions')
    N = Names.shape[0]

    # style use and color 
    plt.style.use('seaborn')
    color_set = sns.color_palette('Paired',n_colors = N + 1)

    # create annotation style
    linewidth = 1.3
    text = 'Start decision-making'
    bbox= dict(boxstyle='round', fc = 'yellow', alpha = 0.3)
    arrowprops = dict(arrowstyle = "->")
    size = 9

    ################# first graph ############## 
    fig, ax = plt.subplots(2, 1, sharex = True)
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)

        # first subplot: number of deaths 
        df.plot(x = 'Date', y = 'Number of new deaths', ax = ax[0], \
                legend = False, fontsize = 10, c=color_set[i])

        # second subplot: VSL
        df.plot(x = 'Date', y = 'Value of statistical life-year (VSL) loss',\
                ax = ax[1], legend = False, fontsize = 10,c=color_set[i])

    ax[0].set_title('Number of new deaths per day')
    # ylabel
    ax[1].set_ylabel("US dollars (in millions)")
    ax[1].set_title('Value of statistical life-year (VSL) loss per day')
    # add thousand sign 
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    

    """# add annotation 
    is_date = df.loc[df['Date']== pd.to_datetime(decision_d)]
    y1 = is_date['Number of new deaths']
    ax[0].annotate(text,(decision_d, y1),xytext = (60, 30), \
                    textcoords='offset points', size = size,\
                    bbox = bbox, arrowprops = arrowprops)
    
    y2 = is_date['Value of statistical life-year (VSL) loss']
    ax[1].annotate(text,(decision_d, y2),xytext = (60, 30), \
                    textcoords='offset points', size = size,\
                    bbox = bbox, arrowprops = arrowprops)"""
    # # adjust space
    # plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
   
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3)
    
    # # add legend
    # ax.flatten()[-1].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.7),fontsize = 10)
    plt.savefig("table-%s-1.png"%(table), dpi = 200)
    plt.close()
    
    ############### second graph #################
    # define figure and axes
    fig, ax = plt.subplots(2, 1, sharex = True)

    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)
        # first subplot: assumpation of unemployment rate and actual unemployment rate
        df.loc[df['Date'] >= day].plot(x = 'Date', y = 'Assumption under selected social distancing',\
                                        ax = ax[0], fontsize = 10, legend = False,  c=color_set[i])

        # second subplot: wage loss over time
        df.loc[df['Date'] >= day].plot(x = 'Date', y = 'Wage loss', title = 'Wage loss per day', \
                                      ax = ax[1], legend = False, fontsize = 10,  c=color_set[i])
                                              

    actual_unemp.plot(x = 'Date', y = 'Actual unemployment rate', ax = ax[0], legend = False,fontsize = 10,marker= '.',\
                      c=color_set[N])
    ax[0].set_title('Unemployment rate \n (Assumption: Assumption for unemployment rate under selected social distancing)')
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    sim_start_d = df['Date'].iloc[0]
    ax[1].set_xlim(left = sim_start_d)
    ax[1].set_ylabel("US dollars (in millions)")
    
    """# add annotation
    is_date = df.loc[df['Date']== pd.to_datetime(self.decision_d)]
    y1 = is_date['Assumption under selected social distancing']
    ax[0].annotate(text,(self.decision_d,y1),xytext= (-40, -30), \
                    textcoords='offset points',size = size,\
                    bbox = bbox, arrowprops = arrowprops)

    y2 = is_date['Wage loss']
    ax[1].annotate(text,(self.decision_d,y2),xytext=  (-90, -30), \
                    textcoords='offset points',size = size,\
                    bbox = bbox, arrowprops = arrowprops)"""
    names.append('Actual data')
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax[0].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, -0.5),fontsize = 10)
    plt.savefig("table-%s-2.png"%(table), dpi = 200)
    plt.close()
    
    ######## third graph #########
    # first subplot: number of diagnosed by testing type
    fig, ax = plt.subplots(3, 1, sharex = True)
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)

        df.loc[df['Date'] >= day].plot(x = 'Date', y = 'number of new diagnosis through universal testing', ax = ax[0], fontsize = 10, legend = False, c=color_set[i])
        df.loc[df['Date'] >= day].plot(x = 'Date', y = 'number of new diagnosis through contact tracing', ax = ax[1], fontsize = 10, legend = False, c=color_set[i])
        df.plot(x = 'Date', y = 'number of new diagnosis through symptom-based testing', ax = ax[2], fontsize = 10,legend = False, c=color_set[i])
       
    # fig.suptitle("Number of new diagnosis through testing type per day")
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[2].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[0].set_title('Number of new diagnosis through universal testing')
    ax[1].set_title('Number of new diagnosis through contact tracing')
    ax[2].set_title('Number of new diagnosis through symptom-based testing')
  
    # plt.subplots_adjust(hspace = 0.4, wspace= 0.3, right = 0.8)
    plt.subplots_adjust(hspace = 0.4, wspace= 0.3)
    # ax.flatten()[-1].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 1.4),fontsize = 10)
    plt.savefig("table-%s-3-1.png"%(table), dpi = 200)
    plt.close()

    # second subplot: 
    fig, ax = plt.subplots(4, 1, sharex = True)
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)

        df.loc[df['Date'] >= day].plot(x = 'Date', y = 'cost of universal testing', ax = ax[0], fontsize = 10, legend = False, c=color_set[i])
        df.loc[df['Date'] >= day].plot(x = 'Date', y = 'cost of contact tracing', ax = ax[1], fontsize = 10, legend = False, c=color_set[i])
        df.plot(x = 'Date', y = 'cost of symptom-based testing', ax = ax[2], fontsize = 10, legend = False, c=color_set[i])
        df.plot(x = 'Date', y = 'total cost of testing', ax = ax[3], fontsize = 10, legend = False, c=color_set[i])
       
    ax[0].set_title('Cost of universal testing')
    ax[1].set_title('Cost of contact tracing')
    ax[2].set_title('Cost of symptom-based testing')
    ax[3].set_title('Total cost of testing')
   
    ax[1].set_ylabel('US dollars (in millions)')
    ax[1].yaxis.set_label_coords(-0.1, -0.07)		
    plt.subplots_adjust(hspace = 0.4, wspace= 0.3, right = 0.8)
    ax.flatten()[-1].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 1.6),fontsize = 10)
    plt.savefig("table-%s-3-2.png"%(table), dpi = 200)
    plt.close()


    ######### forth graph #############
    # plot subplot
    fig, ax = plt.subplots(3, 1, sharex = True)
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)

        # first subplot: contact decision 
        df.plot(x = 'Date', y ='Percent reduction in contacts through social distancing', \
                legend = False, fontsize = 10, ax = ax[0], c=color_set[i],linewidth= linewidth) 

        # second subplot: contact tracing testing decision
        df.plot(x = 'Date', y ='Testing capacity – maximum tests per day through contact tracing', \
                legend = False, fontsize = 10, ax = ax[1], c=color_set[i],linewidth= linewidth) 
        # third subplot universal testing decision 
        df.plot(x = 'Date', y ='Testing capacity – maximum tests per day through universal testing', \
                legend = False, fontsize = 10, ax = ax[2], c=color_set[i],linewidth= linewidth) 
   	
    ax[0].set_ylim(bottom = 0, top = 1)
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax[0].set_ylabel('Proportion')
    ax[0].set_title('Percent reduction in contacts through social distancing')
    
    ax[1].set_ylim(bottom = 0)
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[1].set_ylabel('Testing capacity')
    ax[1].set_title('Testing capacity – maximum tests per day through contact tracing')
    
    ax[2].set_ylim(bottom = 0)
    ax[2].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[2].set_ylabel('Testing capacity')
    ax[2].set_title('Testing capacity – maximum tests per day through universal testing')
    """
    # add annotation
    is_date = df.loc[df['Date']== pd.to_datetime(self.decision_d)]
    y1 = is_date['Percent reduction in contacts through social distancing']
    ax[0].annotate(text,(self.decision_d, y1),xytext=(-90, 30), \
                    textcoords='offset points',size = size,\
                    bbox = bbox, arrowprops = arrowprops)
    
    y2 = is_date['Testing capacity – maximum tests per day through contact tracing']
    ax[1].annotate(text,(self.decision_d, y2),xytext=(-90, 30), \
                    textcoords='offset points',size = size,\
                    bbox = bbox, arrowprops = arrowprops)"""
    
    plt.subplots_adjust(hspace = 0.4, wspace= 0.3, right = 0.8)
    ax.flatten()[-1].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 1.5),fontsize = 10)
    fig.suptitle('User choice')
    plt.savefig("table-%s-4.png"%(table), dpi = 200)
    plt.close()

     ######### fifth graph #############

    # first subplot: cumulative diagnosis
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'simulated cumulative diagnosis', fontsize = 10, ax = ax, c=color_set[i], legend = False)
        
    actual_data.plot(x = 'Date', y = 'actual cumulative diagnosis', fontsize = 10, ax = ax, c=color_set[N], legend = False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax.set_title("Cumulative diagnosis")
    names.append('Actual data')							
    # plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3)
    # ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.4),fontsize = 10)
    plt.savefig("table-%s-5-1.png"%(table), dpi = 200)
    plt.close()

    # second subplot: cumulative deaths
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'simulated cumulative deaths', fontsize = 10, ax = ax, c=color_set[i])
        
    actual_data.plot(x = 'Date', y = 'actual cumulative deaths', fontsize = 10, ax = ax,  c=color_set[N])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax.set_title("Cumulative deaths")
    names.append('Actual data')	
    						
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.4),fontsize = 10)
    plt.savefig("table-%s-5-2.png"%(table), dpi = 200)
    plt.close()

    # third subplot: cumulative hospitalization
    fig, ax = plt.subplots()
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'simulated cumulative hospitalized', fontsize = 10, ax = ax, c=color_set[i], legend = False)
        
    actual_data.plot(x = 'Date', y = 'actual cumulative hospitalized', fontsize = 10, ax = ax, c=color_set[N], legend = False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax.set_title("Cumulative hospitalizations")
    names.append('Actual data')						
    # plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3)
    # ax.legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.4),fontsize = 10)
    plt.savefig("table-%s-5-3.png"%(table), dpi = 200)
    plt.close()


    # forth subplot: number of people with infection 
    fig, ax = plt.subplots(2, 1, sharex = True)
    names = []
    for i in range(N):
        name = Names.loc[i,0]
        names.append(name)
        df = excel.parse(sheet_name = name)
        df.plot(x = 'Date', y = 'number of infected, diagnosed', legend = False, fontsize = 10, ax = ax[0],c=color_set[i])
        df.plot(x = 'Date', y = 'number of infected, undiagnosed', legend = False, fontsize = 10, ax = ax[1], c=color_set[i])
    
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ','))) 
    ax[0].set_title("Number of people with diagnosed infection")
    ax[1].set_title("Number of people with undiagnosed infection")
    						
    plt.subplots_adjust(hspace = 0.2, wspace= 0.3, right = 0.8)
    ax.flatten()[-1].legend(labels = names, loc = 'lower left', bbox_to_anchor=(1, 0.9),fontsize = 10)
    plt.savefig("table-%s-5-4.png"%(table), dpi = 200)
    plt.close()


    #### 

    merge_image_colab(table)