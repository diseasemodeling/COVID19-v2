import pandas as pd

# Function to write read me sheet for excel file
def write_rm():
    para_l = ['Date', 'Value of statistical life-year (VSL) loss',\
               'Number of new deaths', 'Wage loss',\
               'Unemployment rate assumption under selected social distancing',\
               'cost of universal testing' ,'cost of contact tracing',\
               'cost of symptom-based testing','total cost of testing',\
               'number of new diagnosis through contact tracing',\
               'number of new diagnosis through symptom-based testing',\
               'number of new diagnosis through universal testing',\
               'Percent reduction in contacts through social distancing',\
               'Testing capacity – maximum tests per day through contact tracing',\
               'Testing capacity – maximum tests per day through universal testing',\
               'number of infected, undiagnosed', 'number of infected, diagnosed',\
               'simulated cumulative diagnosis','simulated cumulative hospitalized',\
               'simulated cumulative deaths']
    type_para_l = ['Date','Economic impact','Epidemic impact','Economic impact',\
                  'Economic impact','Economic impact','Economic impact','Economic impact',\
                  'Economic impact','Epidemic impact','Epidemic impact','Epidemic impact',\
                  'User decision input','User decision input','User decision input',\
                  'Epidemic impact','Epidemic impact','Epidemic impact','Epidemic impact','Epidemic impact']
    dict_rm = {'Parameters':para_l, 'Parameter type':type_para_l}
    df = pd.DataFrame.from_dict(dict_rm)
    df = df.set_index('Parameters')

    return df





# if __name__ == "__main__":
#     write_rm()