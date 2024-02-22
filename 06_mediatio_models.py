# pip install rpy2==3.5.10


# import numpy as np
import pandas as pd
pd.options.display.max_rows = 1000
from pymatreader import read_mat
# from scipy import stats
# import seaborn as sns
# import matplotlib.pyplot as plt

path = './plots/correlations/'
freqs_low_pass =['40']

# for low_pass in freqs_low_pass:
output_file = './output_mediation/all_iterations_output.csv'


# df_mf = pd.read_csv('./data/data_generated/22_MF_all_subjects_0.5_{}.csv'.format(low_pass))
# df_acw = pd.read_csv('./data/data_generated/12_ACW_all_subjects_05-{}.csv'.format(low_pass))
df_mf = pd.read_csv('./data/data_generated/22_MF_all_subjects_0.5_40.csv')
df_acw = pd.read_csv('./data/data_generated/12_ACW_all_subjects_05-40.csv')
# cols_needed = ['F3','F4','Pz','O1','O2','Oz','subject','trial']
cols_needed = ['F4','O2','subject','trial']
df_mf = df_mf.filter(regex='|'.join(cols_needed))
df_mf = df_mf[df_mf.columns.drop(list(df_mf.filter(regex='POz')))]
df_mf = df_mf[df_mf.columns.drop(list(df_mf.filter(regex='AF')))]
cols_needed = ['5s','subject','trial']
df_mf = df_mf.filter(regex='|'.join(cols_needed))

# cols_needed = ['F3','F4','Pz','O1','O2','Oz','subject','trial']
cols_needed = ['F4','O2','subject','trial']
df_acw = df_acw.filter(regex='|'.join(cols_needed))
df_acw = df_acw[df_acw.columns.drop(list(df_acw.filter(regex='POz')))]
df_acw = df_acw[df_acw.columns.drop(list(df_acw.filter(regex='AF')))]
df_acw = df_acw[df_acw.columns.drop(list(df_acw.filter(regex='acw')))]
cols_needed = ['5s','subject','trial']
df_acw = df_acw.filter(regex='|'.join(cols_needed))


probe1 = read_mat('../data/data_raw/resting_state_thought_probe/resting_probe1.mat')

probe1 = probe1['resting_probe1'].T
probe1 = probe1.reshape((700,1)).tolist()
probe1 = pd.DataFrame(probe1,columns=['probe1'])


df = pd.merge(df_mf,df_acw,how='inner', on=['subject','trial'])
df['probe1'] = probe1['probe1']
dv = 'probe1'

# for chan in ['F3','F4','Pz','O1','O2','Oz']:
for chan in ['F4','O2']:
    for segment in ['5s','15s','25s']:

        md = '{}_mf_{}'.format(chan,segment)
        iv = '{}_acz_{}'.format(chan,segment)
        print('\n\n\n###########################')
        print(iv,'\t',md,'\t',dv)

        independent_variable = df[iv]
        mediator_variable = df[md]
        dependent_variable = df[dv]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        independent_variable_std = scaler.fit_transform(independent_variable.values.reshape(-1, 1))
        mediator_variable_std = scaler.fit_transform(mediator_variable.values.reshape(-1, 1))
        dependent_variable_std = scaler.fit_transform(dependent_variable.values.reshape(-1, 1))

        df_std = pd.DataFrame({
            iv: independent_variable_std.flatten(),
            md: mediator_variable_std.flatten(),
            dv: dependent_variable_std.flatten()})

        df_std = df_std[[iv,md,dv]]
        df_std.columns = ['independent','mediator','dependent']
        df_std=df_std.dropna()

        print(iv.replace('_mf_',''))

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        # Convert the pandas DataFrame to an R data frame
        pandas2ri.activate()
        r_data = pandas2ri.DataFrame(df_std)

        # Set up R environment
        r = robjects.r
        my_string = str(iv.replace('_mf_',''))
        with open(output_file, 'a') as f:
            f.write(my_string+'\n')

        
        # Specify the R code
        r_code = """
        suppressMessages(library(lavaan))
        suppressMessages(library(knitr))
        # suppressMessages(library(broom)) #dependencies tidyr, stringr
        suppressMessages(library(dplyr))

        # Assign the data frame to a variable
        data <- data

        # Define the mediation model
        mod1 <- "# a path
                mediator ~ a * independent
                # b path
                dependent ~ b * mediator
                # c prime path 
                dependent ~ cp * independent
                # indirect and total effects
                ab := a * b
                total := cp + ab"
        model_1 <- sem(mod1, data = data, bootstrap = 5000)
        # print(summary(model_1))
        # conf_intervals <- confint(model_1)
        # print(conf_intervals)
        # # model1
        
        parameter_estimates <- parameterEstimates(model_1)

        # Print parameter estimates with confidence intervals
        print(parameter_estimates)

        """
        # Create an environment for executing the R code
        env = robjects.globalenv
        # print("ab estimates:")
        # print(ab_estimates)
        # input()
        # Assign the R data frame to the specified variable name
        env['data'] = r_data
        # Execute the R code
        robjects.r(r_code)
        # robjects.r(f'write.table(output, file="{output_file}", append=TRUE, sep=",", col.names=FALSE, row.names=FALSE)')
        with open(output_file, 'a') as f:
            f.write('\n\n')

        import statsmodels.api as sm    

        independent_variable_std = df_std['independent']
        mediator_variable_std = df_std['mediator']
        dependent_variable_std = df_std['dependent']

        predictor_matrix = pd.concat([independent_variable_std, mediator_variable_std], axis=1)
        predictor_matrix = sm.add_constant(predictor_matrix)
        model = sm.OLS(dependent_variable_std, predictor_matrix)
        results = model.fit()
        coefficients = results.params
        p_values = results.pvalues
        output = pd.DataFrame({'Coefficient': coefficients, 'P-value': p_values})
        print('SUMMARY\n',output)

        # input()








